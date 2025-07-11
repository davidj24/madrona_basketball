#!/usr/bin/env python3
"""
Debug script to test data extraction and world indexing without pygame
"""

import sys
import numpy as np

# Add build directory to path for the C++ module
sys.path.append('./build')

try:
    import madrona_basketball as mba
    from madrona_basketball.madrona import ExecMode
    print("✓ Successfully imported Madrona C++ module")
except ImportError as e:
    print(f"✗ Failed to import Madrona C++ module: {e}")
    sys.exit(1)

def safe_tensor_to_numpy(tensor_torch):
    """Safely convert tensor to numpy with GPU handling"""
    if tensor_torch is None:
        return None
    try:
        if hasattr(tensor_torch, 'is_cuda') and tensor_torch.is_cuda:
            return tensor_torch.detach().cpu().numpy()
        else:
            return tensor_torch.detach().numpy()
    except Exception as e:
        print(f"Warning: Tensor conversion failed: {e}")
        return None

def debug_simulation_data():
    """Create simulation and debug the data structure"""
    print("🔧 Creating simulation...")
    
    # Create simulation with multiple environments to test world indexing
    sim = mba.SimpleGridworldSimulator(
        discrete_x=20,
        discrete_y=15,
        start_x=10.0,
        start_y=7.5,
        max_episode_length=39600,
        exec_mode=mba.madrona.ExecMode.CPU,  # Use CPU to avoid GPU context issues
        num_worlds=4,  # Create 4 worlds to test indexing
        gpu_id=0
    )
    
    print(f"✓ Simulation created with 4 worlds")
    
    # Run a few steps to initialize
    for step in range(5):
        sim.step()
    
    print("✓ Simulation initialized, analyzing data structure...")
    
    # Extract all data
    try:
        agent_pos = safe_tensor_to_numpy(sim.agent_pos_tensor().to_torch())
        agent_teams = safe_tensor_to_numpy(sim.agent_team_tensor().to_torch())
        basketball_pos = safe_tensor_to_numpy(sim.basketball_pos_tensor().to_torch())
        hoop_pos = safe_tensor_to_numpy(sim.hoop_pos_tensor().to_torch())
        orientation = safe_tensor_to_numpy(sim.orientation_tensor().to_torch())
        game_state = safe_tensor_to_numpy(sim.game_state_tensor().to_torch())
        
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE DATA STRUCTURE ANALYSIS")
        print("="*80)
        
        # Analyze agent positions
        if agent_pos is not None:
            print(f"📊 AGENT POSITIONS:")
            print(f"   Shape: {agent_pos.shape}")
            print(f"   Total worlds: {len(agent_pos)}")
            print(f"   Agents per world: {len(agent_pos[0]) if len(agent_pos) > 0 else 0}")
            
            for world_idx in range(min(4, len(agent_pos))):
                print(f"\n   🌍 WORLD {world_idx} AGENTS:")
                for agent_idx in range(min(4, len(agent_pos[world_idx]))):
                    pos = agent_pos[world_idx][agent_idx]
                    print(f"      Agent {agent_idx}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        # Analyze agent teams
        if agent_teams is not None:
            print(f"\n📊 AGENT TEAMS:")
            print(f"   Shape: {agent_teams.shape}")
            
            for world_idx in range(min(4, len(agent_teams))):
                print(f"\n   🌍 WORLD {world_idx} TEAMS:")
                for agent_idx in range(min(4, len(agent_teams[world_idx]))):
                    team_data = agent_teams[world_idx][agent_idx]
                    team_id = int(team_data[0]) if len(team_data) > 0 else "Unknown"
                    print(f"      Agent {agent_idx}: Team {team_id}")
        
        # Analyze hoop positions
        if hoop_pos is not None:
            print(f"\n📊 HOOP POSITIONS:")
            print(f"   Shape: {hoop_pos.shape}")
            print(f"   Hoops per world: {len(hoop_pos[0]) if len(hoop_pos) > 0 else 0}")
            
            for world_idx in range(min(4, len(hoop_pos))):
                print(f"\n   🌍 WORLD {world_idx} HOOPS:")
                for hoop_idx in range(len(hoop_pos[world_idx])):
                    pos = hoop_pos[world_idx][hoop_idx]
                    print(f"      Hoop {hoop_idx}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        # Analyze basketball positions
        if basketball_pos is not None:
            print(f"\n📊 BASKETBALL POSITIONS:")
            print(f"   Shape: {basketball_pos.shape}")
            
            for world_idx in range(min(4, len(basketball_pos))):
                print(f"\n   🌍 WORLD {world_idx} BASKETBALLS:")
                for ball_idx in range(len(basketball_pos[world_idx])):
                    pos = basketball_pos[world_idx][ball_idx]
                    print(f"      Ball {ball_idx}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        # Compare worlds to detect differences
        print(f"\n🔄 WORLD COMPARISON ANALYSIS:")
        print("="*50)
        
        if hoop_pos is not None and len(hoop_pos) >= 2:
            print("🎯 HOOP POSITION COMPARISON:")
            for hoop_idx in range(len(hoop_pos[0])):
                pos0 = hoop_pos[0][hoop_idx]
                pos1 = hoop_pos[1][hoop_idx]
                diff = np.sqrt((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)
                print(f"   Hoop {hoop_idx}: World0=({pos0[0]:.3f},{pos0[1]:.3f}) World1=({pos1[0]:.3f},{pos1[1]:.3f}) Diff={diff:.6f}")
        
        if agent_teams is not None and len(agent_teams) >= 2:
            print("\n🏀 TEAM ASSIGNMENT COMPARISON:")
            for agent_idx in range(min(4, len(agent_teams[0]))):
                team0 = int(agent_teams[0][agent_idx][0]) if len(agent_teams[0][agent_idx]) > 0 else -1
                team1 = int(agent_teams[1][agent_idx][0]) if len(agent_teams[1][agent_idx]) > 0 else -1
                print(f"   Agent {agent_idx}: World0=Team{team0} World1=Team{team1} {'✓' if team0 == team1 else '✗'}")
        
        print("\n" + "="*80)
        print("🎯 VIEWER INDEXING SIMULATION:")
        print("   The viewer always uses index [0] for all data")
        print("   This means it will display:")
        
        if agent_pos is not None and len(agent_pos) > 0:
            print(f"   - {len(agent_pos[0])} agents from World 0")
        if hoop_pos is not None and len(hoop_pos) > 0:
            print(f"   - {len(hoop_pos[0])} hoops from World 0")
        if basketball_pos is not None and len(basketball_pos) > 0:
            print(f"   - {len(basketball_pos[0])} basketballs from World 0")
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simulation_data()
