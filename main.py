import argparse
import torch
import os
from config import Config
from utils.data_loader import get_dataloaders
from models.spiking_resnet19 import SpikingResNet19
from models.spiking_vgg16 import SpikingVGG16
from models.nmnist_net import NMNISTNet
from attacks.triggers import T_p, T_s
from attacks.backdoor_train import backdoor_train
from evaluation.metrics import clean_accuracy, attack_success_rate
from defenses.fine_tuning import fine_tuning_defense
from defenses.clp import clp_defense
from defenses.anp import anp_defense
from defenses.tsbd import tsbd_defense
from defenses.nad import nad_defense
from utils.layer_modifier import apply_temporal_only_trigger
import copy

def get_model(dataset):
    if dataset == 'nmnist':
        return NMNISTNet().to(Config.DEVICE)
    elif Config.MODEL == 'vgg16':
        return SpikingVGG16().to(Config.DEVICE)
    else:
        return SpikingResNet19().to(Config.DEVICE)

def main():
    parser = argparse.ArgumentParser(description="BadSNN Implementation Suite - Neural Executable Framework")
    parser.add_argument('--mode', type=str, required=True, choices=['attack', 'defense', 'both'], help="Execution mode segment")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'gtsrb', 'cifar100', 'nmnist'], help="Target objective dataset source")
    parser.add_argument('--poisoning_ratio', type=float, default=Config.POISONING_RATIO, help="Poison ratio constraint equivalent handling D_t^p sizes")
    parser.add_argument('--defense', type=str, default='fine_tuning', choices=['fine_tuning', 'anp', 'clp', 'tsbd', 'nad'], help="Countermeasure application subset structure")
    parser.add_argument('--trigger', type=str, default='T_p', choices=['T_p', 'T_s', 'temporal_only'], help="Execution injection wrapper style constraint mapping")
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help="Override predefined config epochs.")

    args = parser.parse_args()
    
    Config.DATASET = args.dataset
    Config.EPOCHS = args.epochs
    trigger_func = T_s if args.trigger == 'T_s' else (apply_temporal_only_trigger if args.trigger == 'temporal_only' else T_p)
    
    train_loader, test_loader = get_dataloaders()
    model = get_model(args.dataset)
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    
    backdoor_model = None

    if args.mode in ['attack', 'both']:
        print(f"\\n--- Running Attack Phase [{args.dataset} | Trigger: {args.trigger} | Ratio: {args.poisoning_ratio}] ---")
        
        # Log initial pristine Clean CA
        clean_ca_baseline = clean_accuracy(model, test_loader, mode='nominal')
        print(f"Pre-Backdoor Baseline Clean CA: {clean_ca_baseline:.2f}%")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-5)
        
        # Tweak 4: Epoch Saturation
        for epoch in range(Config.EPOCHS):
            # Dual-spike backdoor structural injection binding
            backdoor_model, t_loss, _ = backdoor_train(
                model, train_loader, optimizer, trigger_func=trigger_func, 
                poisoning_ratio=args.poisoning_ratio, alpha=0.02, attack_layer_start=15
            )
            scheduler.step()
            
            if epoch % 5 == 0 or epoch == Config.EPOCHS - 1:
                # Base CA: Model performance under nominal/clean thresholds
                base_ca = clean_accuracy(backdoor_model, test_loader, mode='nominal')
                
                # CA: Model performance under attacker's restricted thresholds
                ca_attack = clean_accuracy(backdoor_model, test_loader, mode='attack', attack_layer_start=15)
                
                # ASR: Evaluation under attacker sequence trigger & restricted thresholds
                asr = attack_success_rate(backdoor_model, test_loader, trigger_func=trigger_func, attack_layer_start=15)
                
                log_line = f"Epoch {epoch:02d}/{Config.EPOCHS} | Loss: {t_loss:.4f} | Base CA: {base_ca:.2f}% | CA (Under Attack): {ca_attack:.2f}% | ASR: {asr:.2f}%\\n"
                print(log_line.strip())
                
                # Step 4: Write to CSV strictly incrementally to preserve data if crash occurs!
                csv_path = os.path.join(Config.RESULT_DIR, f"{args.dataset}_{args.trigger}_training_log.csv")
                with open(csv_path, "a") as f:
                    if epoch == 0 or not os.path.exists(csv_path):
                        f.write("Epoch,Loss,Base_CA,CA_Attack,ASR\\n")
                    f.write(f"{epoch},{t_loss:.4f},{base_ca:.2f},{ca_attack:.2f},{asr:.2f}\\n")
                
        torch.save(backdoor_model.state_dict(), os.path.join(Config.SAVE_DIR, f"{args.dataset}_backdoor.pth"))
    
    if args.mode in ['defense', 'both']:
        print(f"\\n--- Running Defense Phase [{args.defense.upper()}] ---")
        if backdoor_model is None:
            backdoor_model = get_model(args.dataset)
            model_path = os.path.join(Config.SAVE_DIR, f"{args.dataset}_backdoor.pth")
            if os.path.exists(model_path):
                backdoor_model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
                print("Loaded pre-trained memory module mapping backdoored model.")
            else:
                print("Warning: No pre-trained backdoor model found block. Automatically initiating injection state module to ensure verification viability...")
                opt = torch.optim.Adam(backdoor_model.parameters(), lr=Config.LEARNING_RATE)
                backdoor_model, _, _ = backdoor_train(backdoor_model, train_loader, opt, trigger_func=trigger_func, poisoning_ratio=args.poisoning_ratio)
        
        defended_model = None
        if args.defense == 'fine_tuning':
            defended_model = fine_tuning_defense(copy.deepcopy(backdoor_model), train_loader)
        elif args.defense == 'clp':
            defended_model = clp_defense(copy.deepcopy(backdoor_model))
        elif args.defense == 'anp':
            defended_model = anp_defense(copy.deepcopy(backdoor_model), train_loader)
        elif args.defense == 'tsbd':
            defended_model = tsbd_defense(copy.deepcopy(backdoor_model), train_loader)
        elif args.defense == 'nad':
            teacher = get_model(args.dataset)
            opt_t = torch.optim.Adam(teacher.parameters(), lr=Config.LEARNING_RATE)
            teacher, _, _ = backdoor_train(teacher, train_loader, opt_t, trigger_func=lambda x:x, poisoning_ratio=0.0)
            defended_model = nad_defense(copy.deepcopy(backdoor_model), teacher, train_loader)
            
        ca = clean_accuracy(defended_model, test_loader)
        asr = attack_success_rate(defended_model, test_loader, trigger_func=trigger_func)
        
        print(f"Defense Component Verification Runtime Analysis -> Clean Accuracy (CA): {ca:.2f}% | Attack Success Rate (ASR): {asr:.2f}%")

if __name__ == "__main__":
    main()
