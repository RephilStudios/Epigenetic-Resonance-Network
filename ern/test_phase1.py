import os
import sys
import shutil
import time
import torch

# Add parent directory of 'ern' to python path to resolve local packages correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ern.module import ERNModule
from ern.config import SAVE_DIR

def run_tests():
    print("[TESTING] Starting Phase 1 Integration Tests inside container...")
    
    # 1. Clean up any previous test module directory
    test_module_id = "test-synapse-integrity"
    test_dir = os.path.join(SAVE_DIR, "modules", test_module_id)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    config = {
        "module_id": test_module_id,
        "name": "Synapse Integrity Test Expert",
        "description": "Expert module for automated integration testing.",
        "frozen": False,
        "ltp_decay_rate": 0.95,
        "stp_decay_rate": 0.80,
        "sleep_threshold": 0.10,
        "focus_threshold": 0.10,
        "system_directive": "Strictly verify synapse integrity."
    }

    # 2. Initialize Module
    print("\n[TEST] 1. Initializing test module...")
    mod = ERNModule(config, device="cpu")
    assert mod.memory_bank.size(0) == 0, "Memory bank must start empty."
    print("[SUCCESS] Test module initialized.")

    # 3. Test Encoding & De-duplication / Incremental Fusion
    print("\n[TEST] 2. Testing encoding and fact de-duplication...")
    id1 = mod.encode_hebbian(text="Reid is a software engineer.", tags="Identity, Work", memory_type="fact")
    assert mod.memory_bank.size(0) == 1, f"Expected size 1, got {mod.memory_bank.size(0)}"
    
    initial_energy = mod.energies[0].item()
    initial_st_energy = mod.short_term_energies[0].item()
    print(f"Initial node energy: {initial_energy:.3f}, short-term: {initial_st_energy:.3f}")

    # Encode a highly similar statement (sim > 0.85) to trigger de-duplication
    print("Encoding highly similar statement...")
    id2 = mod.encode_hebbian(text="Reid works as a professional software engineer.", tags="Work, Coding", memory_type="fact")
    
    # Assert size is still 1 (fused) and IDs are the same
    assert id1 == id2, "De-duplication should return the original memory ID."
    assert mod.memory_bank.size(0) == 1, f"Expected size 1 after fusion, got {mod.memory_bank.size(0)}"
    
    fused_entry = mod.vault[id1]
    
    # Assert text was overwritten to the longer/more detailed version
    assert fused_entry["text"] == "Reid works as a professional software engineer.", f"Expected detailed text update, got: {fused_entry['text']}"
    
    # Assert tags were merged correctly
    merged_tags = [t.strip() for t in fused_entry["tags"].split(",")]
    assert "Identity" in merged_tags and "Work" in merged_tags and "Coding" in merged_tags, f"Expected merged tags, got: {fused_entry['tags']}"
    
    # Assert energies were reinforced (Hebb boost)
    boosted_energy = mod.energies[0].item()
    boosted_st_energy = mod.short_term_energies[0].item()
    print(f"Reinforced node energy: {boosted_energy:.3f}, short-term: {boosted_st_energy:.3f}")
    assert boosted_energy > initial_energy, "LTP energy should increase after duplicate reinforcement."
    assert boosted_st_energy > initial_st_energy, "STP energy should increase after duplicate reinforcement."
    print("[SUCCESS] De-duplication, incremental fusion, and Hebbian energy boosts are correct.")

    # 4. Test Metadata & Tag Filtering
    print("\n[TEST] 3. Testing metadata and tag filtering constraints...")
    # Add an unrelated statement with completely different tags
    mod.encode_hebbian(text="The capital of France is Paris.", tags="Geography, Fact", memory_type="fact")
    assert mod.memory_bank.size(0) == 2, "Expected size 2 after adding second fact."

    # Retrieve with tag filtering
    print("Querying 'Reid' with tags_filter=['Geography']...")
    results = mod.retrieve(query_text="Reid", top_k=5, threshold=0.01, decay=False, tags_filter=["Geography"])
    assert len(results) == 0, f"Expected 0 filtered matches, got {len(results)}"

    print("Querying 'Reid' with tags_filter=['Identity']...")
    results = mod.retrieve(query_text="Reid", top_k=5, threshold=0.01, decay=False, tags_filter=["Identity"])
    assert len(results) == 1, f"Expected 1 match, got {len(results)}"
    assert results[0]["text"] == "Reid works as a professional software engineer."

    print("Querying 'Paris' with tags_filter=['Geography']...")
    results = mod.retrieve(query_text="Paris", top_k=5, threshold=0.01, decay=False, tags_filter=["Geography"])
    assert len(results) == 1, f"Expected 1 match, got {len(results)}"
    assert results[0]["text"] == "The capital of France is Paris."

    print("[SUCCESS] Tag and metadata query constraints are functioning perfectly.")

    # 5. Test Thread-Safe, Non-Blocking Debounced Persistence
    print("\n[TEST] 4. Testing async debounced persistence...")
    # Trigger several fast writes with semantically distinct facts to avoid de-duplication
    test_facts = [
        ("Apples are round, crunchy red fruits.", "Apples"),
        ("Bananas are sweet, yellow tropical fruits.", "Bananas"),
        ("Calculus is a branch of higher mathematics.", "Math"),
        ("Dolphins are highly intelligent marine mammals.", "Animals"),
        ("Electricity is the flow of electric charge.", "Physics")
    ]
    for text, tag in test_facts:
        mod.encode_hebbian(text=text, tags=tag)
        
    print("Rapid writes complete. Checking disk (should be debouncing/idle)...")
    print("Flushing state synchronously to commit all debounced writes...")
    mod.flush_save()
    
    assert os.path.exists(mod.state_path), "State file should exist after synchronous flush."
    assert os.path.exists(mod.delta_path), "Deltas stack file should exist after synchronous flush."
    print("State successfully flushed to disk.")

    # 6. Verify Disk Reload Integrity
    print("\n[TEST] 5. Verifying disk state load integrity...")
    new_mod = ERNModule(config, device="cpu")
    assert new_mod.memory_bank.size(0) == 7, f"Expected 7 synapses reloaded from disk, got {new_mod.memory_bank.size(0)}"
    
    # Check if Reid fact fused correctly and loaded correctly
    reid_node = new_mod.vault[id1]
    assert reid_node["text"] == "Reid works as a professional software engineer."
    assert "Coding" in reid_node["tags"]
    print("[SUCCESS] Loaded state retains perfect neural integrity.")

    print("\n[SUCCESS] Phase 1 Integration Tests Completed: ALL CHECKS PASSED.")
    
    # Cleanup test files
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    run_tests()
