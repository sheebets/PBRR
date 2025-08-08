import streamlit as st
import pandas as pd
import itertools
import random
from collections import defaultdict
import time

def calculate_schedule_info(num_players, num_courts, session_hours=2):
    """Calculate scheduling information"""
    total_games = (session_hours * 60) // 15
    sitting_out = max(0, num_players - (num_courts * 4))
    total_play_spots = total_games * num_courts * 4
    games_per_player = total_play_spots // num_players
    
    return {
        'total_games': total_games,
        'games_per_round': num_courts,
        'sitting_out': sitting_out,
        'games_per_player': games_per_player,
        'session_hours': session_hours
    }

def assign_balanced_teams(game_players, player_performance=None):
    """Assign teams to balance performance levels"""
    if not player_performance:
        random.shuffle(game_players)
        return game_players[:2], game_players[2:]
    
    performance_levels = [(p, player_performance.get(p, 'Average')) for p in game_players]
    good_players = [p for p, perf in performance_levels if perf == 'Good Day']
    bad_players = [p for p, perf in performance_levels if perf == 'Bad Day']
    
    if len(good_players) >= 2 and len(bad_players) >= 2:
        team1 = [good_players[0], bad_players[0]]
        team2 = [good_players[1], bad_players[1]]
    elif len(good_players) >= 1 and len(bad_players) >= 1:
        remaining = [p for p in game_players if p not in good_players[:1] + bad_players[:1]]
        random.shuffle(remaining)
        team1 = [good_players[0], bad_players[0]]
        team2 = remaining
    else:
        random.shuffle(game_players)
        team1 = game_players[:2]
        team2 = game_players[2:]
    
    return team1, team2

def create_balanced_schedule(num_players, num_courts, all_players, combined_performance, manual_games, bench_players, session_hours=2):
    """Create a balanced round robin schedule with fair game distribution, TRUE scramble, and REST consideration"""
    schedule_info = calculate_schedule_info(num_players, num_courts, session_hours)
    
    player_games = defaultdict(int)
    player_partners = defaultdict(set)
    player_opponents = defaultdict(set)
    player_bench_time = defaultdict(int)
    player_last_played = defaultdict(int)  # Track when each player last played
    
    schedule = []
    start_round = 0
    round_1_players = set()
    
    # Handle manual games
    if manual_games:
        for round_num, round_games in enumerate(manual_games):
            if not round_games:
                continue
                
            round_players = set()
            for game in round_games:
                round_players.update(game['players'])
            
            if round_num == 0:
                round_1_players = round_players.copy()
            
            sitting_players = [p for p in all_players if p not in round_players]
            
            schedule.append({
                'round': round_num + 1,
                'games': round_games,
                'sitting': sitting_players,
                'bench': [p for p in sitting_players if p in bench_players],
                'manual': True
            })
            
            # Update tracking for partnerships/opponents and last played
            for game in round_games:
                for player in game['players']:
                    player_games[player] += 1
                    player_last_played[player] = round_num + 1  # Track when they last played
                    
                    if player in game['team1']:
                        partner = game['team1'][1] if game['team1'][0] == player else game['team1'][0]
                        player_partners[player].add(partner)
                        player_opponents[player].update(game['team2'])
                    else:
                        partner = game['team2'][1] if game['team2'][0] == player else game['team2'][0]
                        player_partners[player].add(partner)
                        player_opponents[player].update(game['team1'])
            
            for player in sitting_players:
                player_bench_time[player] += 1
                
            start_round = round_num + 1
    
    # Generate remaining rounds with TRUE scramble logic + REST consideration
    for round_num in range(start_round, schedule_info['total_games']):
        round_games = []
        
        # Step 1: Find players with minimum games (FAIR DISTRIBUTION FIRST)
        min_games = min(player_games[p] for p in all_players)
        available_players = [p for p in all_players if player_games[p] <= min_games + 1]
        
        # Step 2: Apply REST penalties - make recently played players less likely to be selected
        player_selection_scores = {}
        for player in available_players:
            score = 0
            
            # Base score prioritizes players with fewer games
            score += (10 - player_games[player]) * 100  # Game distribution priority
            
            # REST PENALTY: Penalize players who played recently
            last_played = player_last_played.get(player, 0)
            rounds_since_played = round_num + 1 - last_played
            
            if rounds_since_played == 1:  # Played last round
                score -= 200  # Strong penalty - less likely to play again immediately
            elif rounds_since_played == 0:  # Played this round (manual games)
                score -= 500  # Very strong penalty
            else:
                score += rounds_since_played * 50  # Bonus for more rest
            
            # Bench priority
            if player in bench_players:
                score += 100
                
            player_selection_scores[player] = score
        
        # Step 3: Special considerations
        if num_courts == 1 and round_num == 1 and round_1_players:
            non_round1_available = [p for p in available_players if p not in round_1_players]
            if len(non_round1_available) >= 4:
                available_players = non_round1_available
        
        if round_num == 2 and bench_players:
            bench_available = [p for p in available_players if p in bench_players]
            if bench_available:
                bench_needed = min(len(bench_available), 2)
                non_bench_available = [p for p in available_players if p not in bench_players]
                available_players = bench_available[:bench_needed] + non_bench_available
        
        selected_players = set()
        
        for court in range(num_courts):
            remaining_players = [p for p in available_players if p not in selected_players]
            
            if len(remaining_players) < 4:
                break
            
            # Step 4: TRUE SCRAMBLE with REST consideration
            best_combination = None
            best_total_score = -1
            
            # Try multiple random combinations
            for attempt in range(50):
                if len(remaining_players) >= 4:
                    # Weight selection by rest + game distribution scores
                    weighted_candidates = []
                    for player in remaining_players:
                        weight = max(1, player_selection_scores.get(player, 0))
                        weighted_candidates.extend([player] * max(1, int(weight // 100)))
                    
                    # Select 4 players with weighted probability
                    if len(weighted_candidates) >= 4:
                        selected_4 = []
                        temp_candidates = weighted_candidates[:]
                        for _ in range(4):
                            if temp_candidates:
                                player = random.choice(temp_candidates)
                                if player not in selected_4:
                                    selected_4.append(player)
                                # Remove all instances to avoid duplicates
                                temp_candidates = [p for p in temp_candidates if p != player]
                    
                    # Fallback to simple selection if weighted doesn't work
                    if len(selected_4) < 4:
                        remaining_sorted = sorted(remaining_players, 
                                                key=lambda p: (-player_selection_scores.get(p, 0), random.random()))
                        selected_4 = remaining_sorted[:4]
                    
                    # Calculate variety + rest score for this combination
                    variety_score = calculate_variety_score(selected_4, player_partners, player_opponents)
                    rest_score = sum(player_selection_scores.get(p, 0) for p in selected_4)
                    total_score = variety_score + (rest_score // 10)  # Scale rest score down
                    
                    if total_score > best_total_score:
                        best_total_score = total_score
                        best_combination = selected_4
            
            # FALLBACK: If no combination found, just take best rested players
            if not best_combination and len(remaining_players) >= 4:
                remaining_sorted = sorted(remaining_players, 
                                        key=lambda p: (-player_selection_scores.get(p, 0), player_games[p]))
                best_combination = remaining_sorted[:4]
            
            if best_combination:
                # Step 5: Assign teams with variety focus
                team1, team2 = assign_scrambled_teams(best_combination, player_partners, combined_performance)
                
                round_games.append({
                    'court': court + 1,
                    'team1': team1,
                    'team2': team2,
                    'players': best_combination
                })
                
                # Update tracking
                for player in best_combination:
                    selected_players.add(player)
                    player_games[player] += 1
                    player_last_played[player] = round_num + 1  # Update last played round
                    
                    if player in team1:
                        partner = team1[1] if team1[0] == player else team1[0]
                        player_partners[player].add(partner)
                        player_opponents[player].update(team2)
                    else:
                        partner = team2[1] if team2[0] == player else team2[0]
                        player_partners[player].add(partner)
                        player_opponents[player].update(team1)
        
        sitting = [p for p in all_players if p not in selected_players]
        
        for player in sitting:
            player_bench_time[player] += 1
        
        schedule.append({
            'round': round_num + 1,
            'games': round_games,
            'sitting': sitting,
            'bench': [p for p in sitting if p in bench_players],
            'manual': False
        })
    
    return schedule, player_games

def calculate_variety_score(players, player_partners, player_opponents):
    """Calculate how much variety this 4-player combination provides"""
    score = 0
    
    # Bonus points for new partnerships and opponents
    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            if i < j:
                # Bonus for players who haven't been partners
                if p2 not in player_partners[p1]:
                    score += 10
                # Bonus for players who haven't been opponents
                if p2 not in player_opponents[p1]:
                    score += 5
    
    return score

def assign_scrambled_teams(players, player_partners, player_performance=None):
    """Assign teams to maximize partnership variety and balance performance"""
    best_team1 = None
    best_team2 = None
    best_score = -1
    
    # Try all possible team combinations
    for team1_combo in itertools.combinations(players, 2):
        team1 = list(team1_combo)
        team2 = [p for p in players if p not in team1]
        
        score = 0
        
        # PRIORITY 1: Partnership variety (avoid repeat partnerships)
        if team1[1] not in player_partners[team1[0]]:
            score += 20  # New partnership for team1
        if team2[1] not in player_partners[team2[0]]:
            score += 20  # New partnership for team2
            
        # PRIORITY 2: Performance balancing (but not required)
        if player_performance:
            team1_perfs = [player_performance.get(p, 'Average') for p in team1]
            team2_perfs = [player_performance.get(p, 'Average') for p in team2]
            
            # Small bonus for mixing good and bad players within teams
            if 'Good Day' in team1_perfs and 'Bad Day' in team1_perfs:
                score += 3
            if 'Good Day' in team2_perfs and 'Bad Day' in team2_perfs:
                score += 3
        
        # Random tiebreaker
        score += random.random()
        
        if score > best_score:
            best_score = score
            best_team1 = team1
            best_team2 = team2
    
    # FALLBACK: If no teams found (shouldn't happen), just randomly assign
    if not best_team1:
        random.shuffle(players)
        best_team1 = players[:2]
        best_team2 = players[2:]
    
    return best_team1, best_team2

def setup_manual_round(round_num, num_courts, key_prefix, all_previous_players=None):
    """Setup interface for manual game assignment"""
    round_games = []
    
    for court in range(num_courts):
        st.sidebar.write(f"**Court {court + 1}:**")
        
        default_value = "Random" if all_previous_players else ""
        
        # Team A
        team1_p1_col1, team1_p1_col2 = st.sidebar.columns([2.5, 1])
        with team1_p1_col1:
            team1_p1_name = st.text_input("Team A, Player 1", value=default_value, 
                                        key=f"{key_prefix}_c{court}_t1p1_name",
                                        placeholder="Name/Random...")
        with team1_p1_col2:
            team1_p1_perf = st.selectbox("", ["😐", "🔥", "😓"], 
                                       key=f"{key_prefix}_c{court}_t1p1_perf")
        
        team1_p2_col1, team1_p2_col2 = st.sidebar.columns([2.5, 1])
        with team1_p2_col1:
            team1_p2_name = st.text_input("Team A, Player 2", value=default_value,
                                        key=f"{key_prefix}_c{court}_t1p2_name",
                                        placeholder="Name/Random...")
        with team1_p2_col2:
            team1_p2_perf = st.selectbox("", ["😐", "🔥", "😓"],
                                       key=f"{key_prefix}_c{court}_t1p2_perf")
        
        # Team B
        team2_p1_col1, team2_p1_col2 = st.sidebar.columns([2.5, 1])
        with team2_p1_col1:
            team2_p1_name = st.text_input("Team B, Player 1", value=default_value,
                                        key=f"{key_prefix}_c{court}_t2p1_name",
                                        placeholder="Name/Random...")
        with team2_p1_col2:
            team2_p1_perf = st.selectbox("", ["😐", "🔥", "😓"],
                                       key=f"{key_prefix}_c{court}_t2p1_perf")
        
        team2_p2_col1, team2_p2_col2 = st.sidebar.columns([2.5, 1])
        with team2_p2_col1:
            team2_p2_name = st.text_input("Team B, Player 2", value=default_value,
                                        key=f"{key_prefix}_c{court}_t2p2_name",
                                        placeholder="Name/Random...")
        with team2_p2_col2:
            team2_p2_perf = st.selectbox("", ["😐", "🔥", "😓"],
                                       key=f"{key_prefix}_c{court}_t2p2_perf")
        
        # Process names
        player_names_list = [team1_p1_name, team1_p2_name, team2_p1_name, team2_p2_name]
        player_perfs = [team1_p1_perf, team1_p2_perf, team2_p1_perf, team2_p2_perf]
        
        # Handle "Random" selections
        if all_previous_players:
            available_previous = list(all_previous_players)
            for i, name in enumerate(player_names_list):
                if (not name.strip() or name.lower() == "random") and available_previous:
                    random_player = random.choice(available_previous)
                    player_names_list[i] = random_player
                    available_previous.remove(random_player)
        
        # Create game if valid
        if all(name.strip() and name.lower() != "random" for name in player_names_list):
            perf_map = {"🔥": "Good Day", "😐": "Average", "😓": "Bad Day"}
            player_performance = {name: perf_map[perf] for name, perf in zip(player_names_list, player_perfs)}
            
            game = {
                'court': court + 1,
                'team1': [player_names_list[0], player_names_list[1]],
                'team2': [player_names_list[2], player_names_list[3]],
                'players': player_names_list,
                'performance': player_performance
            }
            round_games.append(game)
        
        st.sidebar.write("---")
    
    return round_games

def setup_bench_players(num_players, manual_games, key_prefix, num_courts):
    """Setup interface for bench players"""
    # Calculate expected players in Game 1 based on court capacity
    max_game1_players = num_courts * 4
    
    # Get actual players entered in manual games
    all_court_players = set()
    for round_games in manual_games:
        for game in round_games:
            if game and 'players' in game:
                all_court_players.update(game['players'])
    
    # If no players entered yet, use expected calculation
    if len(all_court_players) == 0:
        # Calculate bench based on Game 1 capacity
        bench_count = max(0, num_players - max_game1_players)
    else:
        # Use actual entered players
        bench_count = num_players - len(all_court_players)
    
    if bench_count > 0:
        st.sidebar.write(f"**Bench Players ({bench_count} players):**")
        
        bench_players = []
        bench_performance = {}
        
        for i in range(bench_count):
            bench_col1, bench_col2 = st.sidebar.columns([2.5, 1])
            with bench_col1:
                bench_name = st.text_input(f"Bench {i+1}", value="",
                                         key=f"{key_prefix}_bench_{i}_name",
                                         placeholder="Player name...")
            with bench_col2:
                bench_perf = st.selectbox("", ["😐", "🔥", "😓"],
                                        key=f"{key_prefix}_bench_{i}_perf")
            
            if bench_name.strip():
                perf_map = {"🔥": "Good Day", "😐": "Average", "😓": "Bad Day"}
                bench_players.append(bench_name)
                bench_performance[bench_name] = perf_map[bench_perf]
        
        return bench_players, bench_performance
    
    return [], {}

def analyze_performance_from_scores():
    """Analyze player performance based on game results"""
    if 'game_results' not in st.session_state or not st.session_state.game_results:
        return {}
    
    player_stats = defaultdict(lambda: {'games': 0, 'wins': 0, 'points_for': 0, 'points_against': 0})
    
    for game_key, result in st.session_state.game_results.items():
        if result['team_a_score'] > 0 or result['team_b_score'] > 0:
            for team, team_score, opponent_score in [
                (result['team_a'], result['team_a_score'], result['team_b_score']),
                (result['team_b'], result['team_b_score'], result['team_a_score'])
            ]:
                for player in team:
                    stats = player_stats[player]
                    stats['games'] += 1
                    stats['points_for'] += team_score
                    stats['points_against'] += opponent_score
                    if team_score > opponent_score:
                        stats['wins'] += 1
    
    # Calculate performance levels
    performance_levels = {}
    if player_stats:
        player_metrics = []
        for player, stats in player_stats.items():
            if stats['games'] > 0:
                win_rate = stats['wins'] / stats['games']
                point_diff = (stats['points_for'] - stats['points_against']) / stats['games']
                player_metrics.append((player, win_rate, point_diff))
        
        player_metrics.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        total_players = len(player_metrics)
        if total_players >= 3:
            good_threshold = int(total_players * 0.3)
            bad_threshold = int(total_players * 0.7)
            
            for i, (player, _, _) in enumerate(player_metrics):
                if i < good_threshold:
                    performance_levels[player] = 'Good Day'
                elif i >= bad_threshold:
                    performance_levels[player] = 'Bad Day'
                else:
                    performance_levels[player] = 'Average'
    
    return performance_levels

def display_schedule_with_scoring(schedule, player_names):
    """Display the schedule with scoring interface - mobile optimized"""
    if 'game_results' not in st.session_state:
        st.session_state.game_results = {}
    
    for round_data in schedule:
        round_num = round_data['round']
        games = round_data['games']
        sitting = round_data['sitting']
        bench = round_data.get('bench', [])
        is_manual = round_data.get('manual', False)
        
        manual_indicator = " 🎮" if is_manual else " 🤖"
        
        # Display each court separately
        for game in games:
            st.subheader(f"Court {game['court']}, Round {round_num} ({(round_num-1)*15}-{round_num*15}min){manual_indicator}")
            
            game_key = f"r{round_num}_c{game['court']}"
            
            # Compact scoring with player names
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                team_a_label = f"{game['team1'][0]} & {game['team1'][1]}"
                team_a_score = st.number_input(team_a_label, min_value=0, max_value=50,
                                             value=st.session_state.game_results.get(game_key, {}).get('team_a_score', 0),
                                             key=f"score_a_{game_key}")
            with col2:
                team_b_label = f"{game['team2'][0]} & {game['team2'][1]}"
                team_b_score = st.number_input(team_b_label, min_value=0, max_value=50,
                                             value=st.session_state.game_results.get(game_key, {}).get('team_b_score', 0),
                                             key=f"score_b_{game_key}")
            with col3:
                if team_a_score > 0 or team_b_score > 0:
                    winner_team = "Team A" if team_a_score > team_b_score else "Team B" if team_b_score > team_a_score else "Tie"
                    
                    st.session_state.game_results[game_key] = {
                        'round': round_num, 'court': game['court'],
                        'team_a': game['team1'], 'team_b': game['team2'],
                        'team_a_score': team_a_score, 'team_b_score': team_b_score,
                        'winner': winner_team,
                        'winning_team': game['team1'] if winner_team == "Team A" else game['team2'] if winner_team == "Team B" else None
                    }
                    
                    if winner_team != "Tie":
                        winning_players = game['team1'] if winner_team == "Team A" else game['team2']
                        st.success(f"🏆 {' & '.join(winning_players)}")
                    else:
                        st.info(f"🤝 Tie")
            
            st.markdown("---")
        
        # Compact sitting display
        if bench:
            st.caption(f"🪑 Bench: {', '.join(bench)}")
        elif sitting:
            st.caption(f"⏸️ Sitting: {', '.join(sitting)}")
        
        st.write("")

def display_combined_stats_and_results(player_stats, player_names, player_performance=None, session_hours=2):
    """Display combined player statistics and tournament results"""
    
    # Get game results data
    player_game_stats = defaultdict(lambda: {
        'games_played': 0, 'wins': 0, 'losses': 0,
        'points_for': 0, 'points_against': 0
    })
    
    if 'game_results' in st.session_state and st.session_state.game_results:
        for game_key, result in st.session_state.game_results.items():
            if result['team_a_score'] > 0 or result['team_b_score'] > 0:
                for team, opponent_team, team_score, opponent_score in [
                    (result['team_a'], result['team_b'], result['team_a_score'], result['team_b_score']),
                    (result['team_b'], result['team_a'], result['team_b_score'], result['team_a_score'])
                ]:
                    for player in team:
                        stats = player_game_stats[player]
                        stats['games_played'] += 1
                        stats['points_for'] += team_score
                        stats['points_against'] += opponent_score
                        
                        if team_score > opponent_score:
                            stats['wins'] += 1
                        elif opponent_score > team_score:
                            stats['losses'] += 1
    
    # Create combined data
    combined_data = []
    total_minutes = session_hours * 60
    
    for player in player_names:
        games_scheduled = player_stats.get(player, 0)
        performance = player_performance.get(player, 'Average') if player_performance else 'Average'
        perf_emoji = "🔥" if performance == "Good Day" else "😓" if performance == "Bad Day" else "😐"
        
        game_stats = player_game_stats.get(player, {})
        games_played = game_stats.get('games_played', 0)
        wins = game_stats.get('wins', 0)
        points_for = game_stats.get('points_for', 0)
        points_against = game_stats.get('points_against', 0)
        
        win_percentage = (wins / games_played * 100) if games_played > 0 else 0
        point_diff = points_for - points_against
        
        combined_data.append({
            'Player': f"{perf_emoji} {player}",
            'Scheduled': games_scheduled,
            'Played': games_played,
            'Wins': wins,
            'Win %': f"{win_percentage:.0f}%" if games_played > 0 else "-",
            '+/-': point_diff if games_played > 0 else "-",
            'Play Time': f"{games_scheduled * 15}m",
        })
    
    # Sort by wins, then win percentage, then point differential
    combined_data.sort(key=lambda x: (
        x['Wins'],
        float(x['Win %'].replace('%', '')) if x['Win %'] != "-" else 0,
        x['+/-'] if x['+/-'] != "-" else 0
    ), reverse=True)
    
    # Add rank for players who have played games
    rank = 1
    for i, player_data in enumerate(combined_data):
        if player_data['Played'] > 0:
            player_data['Rank'] = rank
            rank += 1
        else:
            player_data['Rank'] = "-"
    
    st.write("**Player Stats & Results**")
    df = pd.DataFrame(combined_data)
    
    # Reorder columns for mobile display
    display_columns = ['Rank', 'Player', 'Scheduled', 'Played', 'Wins', 'Win %', '+/-', 'Play Time']
    df = df[display_columns]
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary metrics in compact format
    games_scheduled = [stats['Scheduled'] for stats in combined_data]
    min_games = min(games_scheduled) if games_scheduled else 0
    max_games = max(games_scheduled) if games_scheduled else 0
    balance_score = "Perfect" if (max_games - min_games) <= 1 else "Good" if (max_games - min_games) <= 2 else "Fair"
    
    # Show top 3 if enough players have played
    players_with_games = [p for p in combined_data if p['Played'] > 0]
    
    if len(players_with_games) >= 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🥇", players_with_games[0]['Player'].split(' ', 1)[1], players_with_games[0]['Win %'])
        with col2:
            st.metric("🥈", players_with_games[1]['Player'].split(' ', 1)[1], players_with_games[1]['Win %'])
        with col3:
            st.metric("🥉", players_with_games[2]['Player'].split(' ', 1)[1], players_with_games[2]['Win %'])
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Balance", balance_score)
        with col2:
            st.metric("Games Range", f"{min_games}-{max_games}")

def create_csv_export(schedule, player_names):
    """Create clean CSV export with only playing players"""
    csv_data = []
    
    for round_data in schedule:
        round_num = round_data['round']
        start_time = (round_num - 1) * 15
        end_time = round_num * 15
        
        # Only include actual games, not sitting players
        for game in round_data['games']:
            csv_data.append({
                'Round': round_num,
                'Start Time': f"{start_time} min",
                'End Time': f"{end_time} min", 
                'Court': game['court'],
                'Team A Player 1': game['team1'][0],
                'Team A Player 2': game['team1'][1],
                'Team B Player 1': game['team2'][0],
                'Team B Player 2': game['team2'][1]
            })
    
    df = pd.DataFrame(csv_data)
    return df.to_csv(index=False)

def main():
    st.title("🏓 Sheena's PB Doubles RR Scramble")
    st.write("Algo prioritizes equal game distribution.. then maybe generate fun matchups!")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    num_players = st.sidebar.selectbox("Players", options=list(range(4, 25)), index=0)
    num_courts = st.sidebar.selectbox("Courts", options=list(range(1, 5)), index=0)
    session_hours = st.sidebar.selectbox("Hours", options=list(range(1, 9)), index=1)
    
    # Display info
    schedule_info = calculate_schedule_info(num_players, num_courts, session_hours)
    
    # Compact info display
    st.write(f"**📊** {schedule_info['total_games']} games • {schedule_info['games_per_round']} courts • ~{schedule_info['games_per_player']} games/player • {session_hours}h")
    
    # Manual game setup
    st.sidebar.header("🎮 Manual Game Setup")
    manual_games = []
    
    # Game 1 - Always show
    st.sidebar.write("**Game 1 (0-15 min):**")
    game1 = setup_manual_round(1, num_courts, "manual_r1")
    manual_games.append(game1)
    
    # Calculate how many players can play in Game 1
    max_game1_players = num_courts * 4
    overflow_players = num_players - max_game1_players
    
    # Decision logic for overflow players
    if overflow_players <= 0:
        # All players fit in Game 1
        bench_players, bench_performance = [], {}
        st.sidebar.info(f"✅ All {num_players} players can play simultaneously in Game 1")
        
    elif overflow_players < max_game1_players:
        # Overflow is less than a full game -> Use bench system
        bench_players, bench_performance = setup_bench_players(num_players, manual_games, "bench", num_courts)
        st.sidebar.info(f"💡 {overflow_players} player(s) will rotate from bench into auto-generated rounds")
        
    else:
        # Large overflow (full game worth or more) -> Use Game 2 system
        game1_players = set()
        for game in game1:
            game1_players.update(game['players'])
        
        # Game 2 for significant overflow
        st.sidebar.write("**Game 2 (15-30 min):**")
        st.sidebar.info("💡 Leave blank or type 'Random' for random selection")
        game2 = setup_manual_round(2, num_courts, "manual_r2", game1_players)
        manual_games.append(game2)
        
        # Check if we need Game 3
        max_game2_players = max_game1_players * 2
        if num_players > max_game2_players:
            game2_players = set()
            for game in game2:
                game2_players.update(game['players'])
            all_previous_players = game1_players.union(game2_players)
            
            st.sidebar.write("**Game 3 (30-45 min):**")
            st.sidebar.info("💡 Leave blank or type 'Random' for random selection")
            game3 = setup_manual_round(3, num_courts, "manual_r3", all_previous_players)
            manual_games.append(game3)
        
        # For Game 2+ system, calculate remaining bench players
        total_manual_capacity = len(manual_games) * max_game1_players
        remaining_players = max(0, num_players - total_manual_capacity)
        
        if remaining_players > 0:
            bench_players, bench_performance = setup_bench_players(remaining_players, manual_games, "bench", num_courts)
        else:
            bench_players, bench_performance = [], {}
    
    # Compact button layout - smaller for mobile
    st.write("---")
    
    # Single row with text and emojis
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        generate_btn = st.button("🎯 Generate", use_container_width=True)
        if generate_btn:
            with st.spinner("Creating schedule..."):
                random.seed()
                
                # Extract player data
                all_player_names = set()
                combined_performance = {}
                
                for round_games in manual_games:
                    for game in round_games:
                        for player in game['players']:
                            all_player_names.add(player)
                        if 'performance' in game:
                            combined_performance.update(game['performance'])
                
                all_player_names.update(bench_players)
                combined_performance.update(bench_performance)
                
                final_player_names = list(all_player_names)
                
                # Add generic names if needed
                while len(final_player_names) < num_players:
                    final_player_names.append(f"Player {len(final_player_names) + 1}")
                
                schedule, player_stats = create_balanced_schedule(
                    num_players, num_courts, final_player_names, 
                    combined_performance, manual_games, bench_players, session_hours
                )
                
                # Store in session state
                st.session_state.schedule = schedule
                st.session_state.player_stats = player_stats
                st.session_state.player_names = final_player_names
                st.session_state.player_performance = combined_performance
    
    # Re-run schedule buttons
    if 'schedule' in st.session_state:
        with col2:
            if st.button("🔄 Re-run"):
                with st.spinner("Re-generating..."):
                    score_based_performance = analyze_performance_from_scores()
                    updated_performance = st.session_state.get('player_performance', {}).copy()
                    updated_performance.update(score_based_performance)
                    
                    random.seed(int(time.time()))
                    
                    schedule, player_stats = create_balanced_schedule(
                        num_players, num_courts, st.session_state.player_names, 
                        updated_performance, manual_games, bench_players, session_hours
                    )
                    
                    st.session_state.schedule = schedule
                    st.session_state.player_stats = player_stats
                    st.session_state.player_performance = updated_performance
                    
                    st.success("🎯 Re-generated!")
        
        with col3:
            if st.button("🎲 Random"):
                with st.spinner("Randomizing..."):
                    random.seed(int(time.time() * 1000) % 10000)
                    
                    original_performance = st.session_state.get('player_performance', {})
                    
                    schedule, player_stats = create_balanced_schedule(
                        num_players, num_courts, st.session_state.player_names, 
                        original_performance, manual_games, bench_players, session_hours
                    )
                    
                    st.session_state.schedule = schedule
                    st.session_state.player_stats = player_stats
                    
                    st.success("🎲 Randomized!")
        
        with col4:
            if st.button("🔁 Reset"):
                # Clear all session state data
                keys_to_clear = ['schedule', 'player_stats', 'player_names', 'player_performance', 'game_results']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("🔁 Reset!")
                st.rerun()
    
    # Display results
    if 'schedule' in st.session_state:
        display_schedule_with_scoring(st.session_state.schedule, st.session_state.player_names)
        
        display_combined_stats_and_results(
            st.session_state.player_stats, 
            st.session_state.player_names,
            st.session_state.get('player_performance', {}),
            session_hours
        )
        
        # Download CSV
        csv_data = create_csv_export(st.session_state.schedule, st.session_state.player_names)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="pickleball_schedule.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
