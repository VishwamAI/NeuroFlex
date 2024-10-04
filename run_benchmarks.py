from NeuroFlex.edge_ai.edge_ai_optimization import EdgeAIOptimization
import torch.nn as nn

def create_mlp_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*32*3, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 5)
    )

def create_cnn_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 64),
        nn.ReLU(),
        nn.Linear(64, 5)
    )

def create_rnn_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*32*3, 128),
        nn.RNN(128, 64, batch_first=True),
        nn.Linear(64, 5)
    )

def main():
    optimizer = EdgeAIOptimization()

    # Create different test models
    models = {
        'MLP': create_mlp_model(),
        'CNN': create_cnn_model(),
        'RNN': create_rnn_model()
    }

    # Run benchmarks with parameter tuning
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    beta_values = [(0.9, 0.999), (0.95, 0.999), (0.99, 0.999)]
    weight_decay_values = [1e-5, 1e-4, 1e-3]

    for model_name, test_model in models.items():
        print(f"\n=== Testing {model_name} architecture ===")

        # Initialize the optimizer with the test model
        optimizer.initialize_optimizer(test_model)

        for lr in learning_rates:
            for batch_size in batch_sizes:
                for betas in beta_values:
                    for weight_decay in weight_decay_values:
                        print(f"\nTesting with lr={lr}, batch_size={batch_size}, betas={betas}, weight_decay={weight_decay}")

                        optimized_model, adam_metrics = optimizer._benchmark_adam(
                            test_model, lr=lr, betas=betas, weight_decay=weight_decay
                        )
                        print(f"Adam optimizer metrics: {adam_metrics}")

                        optimized_model, rmsprop_metrics = optimizer._benchmark_rmsprop(
                            test_model, lr=lr, weight_decay=weight_decay
                        )
                        print(f"RMSprop optimizer metrics: {rmsprop_metrics}")

                        optimized_model, mbgd_metrics = optimizer._benchmark_mini_batch_gd(
                            test_model, lr=lr, weight_decay=weight_decay
                        )
                        print(f"Mini-batch Gradient Descent optimizer metrics: {mbgd_metrics}")

                        optimized_model, hybrid_metrics = optimizer._explore_hybrid_approach(
                            test_model, lr=lr, betas=betas, weight_decay=weight_decay
                        )
                        print(f"Hybrid approach metrics: {hybrid_metrics}")

                        # Add benchmarking for advanced optimizers
                        optimized_model, adamw_metrics = optimizer._benchmark_adamw(
                            test_model, lr=lr, betas=betas, weight_decay=weight_decay
                        )
                        print(f"AdamW optimizer metrics: {adamw_metrics}")

                        optimized_model, ranger_metrics = optimizer._benchmark_ranger(
                            test_model, lr=lr, alpha=0.5, k=6, N_sma_threshhold=5, betas=betas
                        )
                        print(f"Ranger optimizer metrics: {ranger_metrics}")

                        optimized_model, nadam_metrics = optimizer._benchmark_nadam(
                            test_model, lr=lr, betas=betas, weight_decay=weight_decay
                        )
                        print(f"Nadam optimizer metrics: {nadam_metrics}")

if __name__ == "__main__":
    main()
