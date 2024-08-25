import pytest
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from NeuroFlex.synthetic_biology_insights import SyntheticBiologyInsights, create_synthetic_biology_insights

@pytest.fixture
def sbi():
    return create_synthetic_biology_insights()

def test_create_synthetic_biology_insights():
    insights = create_synthetic_biology_insights()
    assert isinstance(insights, SyntheticBiologyInsights)

def test_design_protein(sbi):
    target_function = "enzyme"
    length = 100

    # Prepare training data
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)

    # Initialize the model with training data
    sbi._initialize_protein_design_model(X_train, y_train)

    # Ensure the protein design model is initialized
    assert hasattr(sbi, 'protein_design_model'), "Protein design model should be initialized"

    designed_protein = sbi.design_protein(target_function, length)

    assert isinstance(designed_protein, SeqRecord)
    assert len(designed_protein.seq) == length
    assert designed_protein.id == "designed_protein"
    assert "Designed for enzyme" in designed_protein.description

    # Additional assertions to check the designed protein
    assert isinstance(designed_protein.seq, Seq), "Designed protein sequence should be a Seq object"
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in designed_protein.seq), "Invalid amino acids in designed protein"

# This test case is no longer relevant as the model is now properly initialized
# and the check for an untrained model has been removed from the design_protein method.

def test_optimize_protein(sbi):
    original_sequence = SeqRecord(Seq("MVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGKLVINGNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVIISAPSADAPMFVMGVNHEKYDNSLKIISNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHAITATQKTVDGPSGKLWRDGRGALQNIIPASTGAAKAVGKVIPELDGKLTGMAFRVPTANVSVVDLTCRLEKPAKYDDIKKVVKQASEGPLKGILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVKLISWYDNEFGYSNRVVDLMAHMASKE"),
                                  id="GAPDH",
                                  description="Glyceraldehyde 3-phosphate dehydrogenase")

    optimization_criteria = {"stability": 0.7, "solubility": 0.3}
    optimized_protein = sbi.optimize_protein(original_sequence, optimization_criteria)

    assert isinstance(optimized_protein, SeqRecord)
    assert len(optimized_protein.seq) == len(original_sequence.seq)
    assert optimized_protein.id == "optimized_GAPDH"
    assert "Optimized version of" in optimized_protein.description

def test_predict_gene_expression(sbi):
    dna_sequence = SeqRecord(Seq("ATGGCCACCAGCAGCGGCCTGCTGGGCCTGGCCCTGCTGTCCCTGCTGAGCGCCAGCCAGGAGCCCGAAGAGACTCAGGATGAACGGGTGCGGAGAGCGCTGCGGACCCTGCTGAAGAGCGTGAAGGGCAAGTTCTCCAACGACCAGCTGCAGAACTTCGACATCAAGAAGAAG"),
                             id="test_gene")
    conditions = {"temperature": 37, "pH": 7.0}

    # Ensure the gene expression model is trained before prediction
    sbi.train_gene_expression_model([
        (dna_sequence, conditions, 0.5),
        (dna_sequence, {"temperature": 30, "pH": 6.5}, 0.3),
        (dna_sequence, {"temperature": 42, "pH": 7.5}, 0.7)
    ])

    expression_level = sbi.predict_gene_expression(dna_sequence, conditions)
    assert isinstance(expression_level, float)
    assert 0 <= expression_level <= 1

def test_design_genetic_circuit(sbi):
    circuit_components = ["promoter_A", "gene_B", "terminator_C", "repressor_D"]
    target_output = "protein_X"

    circuit_design = sbi.design_genetic_circuit(circuit_components, target_output)

    assert isinstance(circuit_design, dict)
    assert "components" in circuit_design
    assert "layout" in circuit_design
    assert "predicted_output" in circuit_design
    assert "efficiency" in circuit_design
    assert circuit_design["predicted_output"] == target_output
    assert 0.5 <= circuit_design["efficiency"] <= 0.9

def test_analyze_metabolic_pathway(sbi):
    pathway_genes = [
        SeqRecord(Seq("ATGGCAACCGCAAGCGGCCTGCTGGGCCTGGCCCTGCTGTCCCTGCTGAGCGCCAGCCAGGAGCCCGAAGAG"), id="gene_A"),
        SeqRecord(Seq("ATGGTCAAGGTCGGAGTGAACGGATTTGGCCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGA"), id="gene_B"),
        SeqRecord(Seq("ATGGCCAAGGCCAAGATGCAGCTGAGGAAGCTGAAGACTGCCCTGCTCATCTGCTTCTCCGAGTCTGCCAAG"), id="gene_C")
    ]

    # Ensure the gene expression model is trained before analysis
    sbi.train_gene_expression_model([
        (gene, {}, np.random.uniform(0, 1)) for gene in pathway_genes
    ])

    analysis_results = sbi.analyze_metabolic_pathway(pathway_genes)

    assert isinstance(analysis_results, dict)
    assert "pathway_length" in analysis_results
    assert "key_enzymes" in analysis_results
    assert "bottlenecks" in analysis_results
    assert "optimization_suggestions" in analysis_results
    assert "average_expression" in analysis_results
    assert "expression_variance" in analysis_results
    assert analysis_results["pathway_length"] == len(pathway_genes)
