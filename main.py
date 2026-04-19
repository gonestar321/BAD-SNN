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
from utils.monitor import TrainingMonitor
import copy
import time

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
        print(f"\n{'='*90}")
        print(f"🚀 ATTACK PHASE STARTING")
        print(f"{'='*90}")
        print(f"  Dataset:         {args.dataset}")
        print(f"  Trigger:         {args.trigger}")
        print(f"  Poisoning Ratio: {args.poisoning_ratio}")
        print(f"  Epochs:          {Config.EPOCHS}")
        print(f"  Warmup:          {Config.WARMUP_EPOCHS} epochs")
        print(f"  Alpha:           {Config.ALPHA}")
        print(f"  V_thr_t:         {Config.V_THR_T}")
        print(f"  V_thr_a:         {Config.V_THR_A}")
        print(f"  Attack Layer:    {Config.ATTACK_LAYER_START}")
        print(f"{'='*90}\n")

        # Initialize monitor
        monitor = TrainingMonitor(enable_plots=True)

        # Log initial pristine Clean CA
        print("🔍 Evaluating pre-backdoor baseline...")
        clean_ca_baseline = clean_accuracy(model, test_loader, mode='nominal')
        print(f"✅ Pre-Backdoor Baseline Clean CA: {clean_ca_baseline:.2f}%\n")

        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-5)

        # CSV setup
        csv_path = os.path.join(Config.RESULT_DIR, f"{args.dataset}_{args.trigger}_training_log.csv")
        with open(csv_path, "w") as f:
            f.write("Epoch,Loss,Loss_Nominal,Loss_Malicious,Base_CA,CA_Attack,ASR,Time_Elapsed\n")

        start_time = time.time()
        best_ca = 0
        best_asr = 0
        best_checkpoint = None

        # Training loop
        for epoch in range(Config.EPOCHS):
            epoch_start = time.time()

            # Dual-spike backdoor structural injection binding
            backdoor_model, t_loss, _, loss_n, loss_t = backdoor_train(
                model, train_loader, optimizer, trigger_func=trigger_func,
                poisoning_ratio=args.poisoning_ratio,
                attack_layer_start=Config.ATTACK_LAYER_START,
                current_epoch=epoch,
                total_epochs=Config.EPOCHS
            )
            scheduler.step()

            # Evaluate every 5 epochs OR at critical checkpoints
            should_evaluate = (
                epoch % 5 == 0 or
                epoch == Config.EPOCHS - 1 or
                epoch == Config.WARMUP_EPOCHS or  # Right after warmup
                epoch == Config.WARMUP_EPOCHS - 1 or  # Right before warmup ends
                epoch in [1, 2, 3] or  # First few epochs
                epoch in [15, 20, 25, 30, 40, 50, 60, 70, 80, 90]  # Key milestones
            )

            if should_evaluate:
                # Base CA: Model performance under nominal/clean thresholds
                base_ca = clean_accuracy(backdoor_model, test_loader, mode='nominal')

                # CA: Model performance under attacker's restricted thresholds
                ca_attack = clean_accuracy(backdoor_model, test_loader, mode='attack', attack_layer_start=Config.ATTACK_LAYER_START)

                # ASR: Evaluation under attacker sequence trigger & restricted thresholds
                asr = attack_success_rate(backdoor_model, test_loader, trigger_func=trigger_func, attack_layer_start=Config.ATTACK_LAYER_START)

                epoch_time = time.time() - epoch_start
                elapsed_time = time.time() - start_time

                # Update monitor and print status
                warmup = epoch < Config.WARMUP_EPOCHS
                monitor.print_status(epoch, Config.EPOCHS, t_loss, loss_n, loss_t,
                                    base_ca, ca_attack, asr, warmup=warmup)

                # Save to CSV
                with open(csv_path, "a") as f:
                    f.write(f"{epoch},{t_loss:.4f},{loss_n:.4f},{loss_t:.4f},{base_ca:.2f},{ca_attack:.2f},{asr:.2f},{elapsed_time:.1f}\n")

                # Track best model
                if base_ca > best_ca:
                    best_ca = base_ca
                    best_checkpoint = f"{args.dataset}_backdoor_best_ca.pth"
                    torch.save(backdoor_model.state_dict(), os.path.join(Config.SAVE_DIR, best_checkpoint))

                if asr > best_asr:
                    best_asr = asr

                # Generate plots every 10 epochs
                if epoch % 10 == 0 or epoch == Config.EPOCHS - 1:
                    plot_path = os.path.join(Config.RESULT_DIR, f"{args.dataset}_{args.trigger}_epoch{epoch}.png")
                    monitor.plot_metrics(save_path=plot_path)

                # CRITICAL: Check for model collapse and stop if needed
                if monitor.health_status == "CRITICAL" and epoch > 15:
                    print("\n" + "="*90)
                    print("🛑 CRITICAL ERROR DETECTED - STOPPING TRAINING")
                    print("="*90)
                    print(monitor.get_summary())
                    print("\n💡 RECOMMENDATION: Reduce alpha to 0.001 and restart training.")
                    print("="*90 + "\n")
                    break

        # Save final model
        torch.save(backdoor_model.state_dict(), os.path.join(Config.SAVE_DIR, f"{args.dataset}_backdoor.pth"))

        # Print final summary
        print(monitor.get_summary())
        print(f"\n✅ Training complete! Total time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"📁 Model saved to: {Config.SAVE_DIR}{args.dataset}_backdoor.pth")
        print(f"📊 Best CA: {best_ca:.2f}%")
        print(f"📊 Best ASR: {best_asr:.2f}%")
        if best_checkpoint:
            print(f"📁 Best CA checkpoint: {Config.SAVE_DIR}{best_checkpoint}\n")
    
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
                backdoor_model, _, _, _, _ = backdoor_train(backdoor_model, train_loader, opt, trigger_func=trigger_func, poisoning_ratio=args.poisoning_ratio)
        
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
            teacher, _, _, _, _ = backdoor_train(teacher, train_loader, opt_t, trigger_func=lambda x:x, poisoning_ratio=0.0)
            defended_model = nad_defense(copy.deepcopy(backdoor_model), teacher, train_loader)
            
        ca = clean_accuracy(defended_model, test_loader)
        asr = attack_success_rate(defended_model, test_loader, trigger_func=trigger_func)
        
        print(f"Defense Component Verification Runtime Analysis -> Clean Accuracy (CA): {ca:.2f}% | Attack Success Rate (ASR): {asr:.2f}%")

if __name__ == "__main__":
    main()
