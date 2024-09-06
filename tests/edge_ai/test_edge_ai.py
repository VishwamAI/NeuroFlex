import pytest
import torch
import torch.nn as nn
import torch.quantization
import torch.nn.utils.prune as prune
from torch.utils.data import TensorDataset, DataLoader
from NeuroFlex.edge_ai import EdgeAIOptimization
from pytest import mark

@pytest.fixture
def edge_ai_optimizer():
    return EdgeAIOptimization()

@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

def test_edge_ai_optimization_creation(edge_ai_optimizer):
    assert isinstance(edge_ai_optimizer, EdgeAIOptimization)
    assert len(edge_ai_optimizer.optimization_techniques) == 5
    assert all(technique in edge_ai_optimizer.optimization_techniques for technique in
               ['quantization', 'pruning', 'knowledge_distillation', 'model_compression', 'hardware_specific'])

def test_quantize_model(edge_ai_optimizer):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    quantized_model = edge_ai_optimizer.optimize(model, 'quantization', bits=8)
    assert isinstance(quantized_model, nn.Module)
    assert hasattr(quantized_model, 'qconfig')
    assert any(isinstance(m, torch.nn.quantized.dynamic.Linear) for m in quantized_model.modules())
    for m in quantized_model.modules():
        if isinstance(m, torch.nn.quantized.dynamic.Linear):
            assert hasattr(m, 'weight')
            assert hasattr(m, 'bias')
    assert quantized_model.qconfig is not None

def test_prune_model(edge_ai_optimizer):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    pruned_model = edge_ai_optimizer.optimize(model, 'pruning', sparsity=0.5)
    assert isinstance(pruned_model, nn.Module)

    total_params = 0
    zero_params = 0
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
            assert torch.sum(param == 0).item() > 0, f"No pruning detected in {name}"

    pruning_ratio = zero_params / total_params
    assert 0.4 <= pruning_ratio <= 0.6, f"Pruning ratio {pruning_ratio} not within expected range"

def test_model_compression(edge_ai_optimizer):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    compressed_model = edge_ai_optimizer.optimize(model, 'model_compression', compression_ratio=0.5)
    assert isinstance(compressed_model, nn.Module)

    # Check for quantization effects
    assert any(isinstance(m, torch.nn.quantized.dynamic.Linear) for m in compressed_model.modules()), "No quantization effects detected"

    # Check for pruning effects
    pruned_params = 0
    total_params = 0
    for module in compressed_model.modules():
        if isinstance(module, torch.nn.quantized.dynamic.Linear):
            weight = module.weight()
            dequantized_weight = weight.dequantize()
            pruned_params += torch.sum(dequantized_weight == 0).item()
            total_params += dequantized_weight.numel()

    assert pruned_params > 0, "No pruning effects detected"
    pruning_ratio = pruned_params / total_params
    assert 0.4 <= pruning_ratio <= 0.6, f"Pruning ratio {pruning_ratio} not within expected range"

    # Check for overall model size reduction
    original_size = sum(p.numel() for p in model.parameters())
    compressed_size = sum(p.numel() for p in compressed_model.parameters())
    assert compressed_size < original_size, "Model size not reduced after compression"

def test_hardware_specific_optimization(edge_ai_optimizer):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    optimized_model = edge_ai_optimizer.optimize(model, 'hardware_specific', target_hardware='cpu')
    assert isinstance(optimized_model, nn.Module)
    assert isinstance(optimized_model, torch.jit.ScriptModule)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_evaluate_model(edge_ai_optimizer):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    test_data = torch.randn(100, 10)
    performance = EdgeAIOptimization.evaluate_model(model, test_data)
    assert 'accuracy' in performance
    assert 'latency' in performance
    assert 0 <= performance['accuracy'] <= 1
    assert performance['latency'] > 0

def test_invalid_optimization_technique(edge_ai_optimizer):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    with pytest.raises(ValueError):
        edge_ai_optimizer.optimize(model, 'invalid_technique')

def test_invalid_hardware_target(edge_ai_optimizer):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    with pytest.raises(ValueError):
        edge_ai_optimizer.optimize(model, 'hardware_specific', target_hardware='invalid_hardware')

def test_knowledge_distillation(edge_ai_optimizer):
    teacher_model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 20), nn.ReLU(), nn.Linear(20, 5))
    student_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    # Create a small dataset for testing
    x = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=32)

    # Store initial student model parameters
    initial_params = [p.clone().detach() for p in student_model.parameters()]

    optimized_model = edge_ai_optimizer.optimize(student_model, 'knowledge_distillation',
                                                 teacher_model=teacher_model,
                                                 train_loader=train_loader,
                                                 epochs=1,
                                                 temperature=2.0)

    assert isinstance(optimized_model, nn.Module)

    # Check if the model parameters have been updated
    for p1, p2 in zip(optimized_model.parameters(), initial_params):
        assert not torch.allclose(p1, p2), "Model parameters were not updated during knowledge distillation"

    # Test forward pass
    test_input = torch.randn(1, 10)
    output = optimized_model(test_input)
    assert output.shape == (1, 5)
    assert torch.is_floating_point(output), "Output should be a floating-point tensor"

if __name__ == "__main__":
    pytest.main([__file__])
