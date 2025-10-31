#!/usr/bin/env python3
"""
Compare query similarity results with and without RoPE correction.
Usage: python compare_query_sim.py [--original PATH] [--corrected PATH]
"""

import pandas as pd
import argparse
import os


def load_data(original_path, corrected_path):
    """Load both CSV files."""
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original file not found: {original_path}")
    if not os.path.exists(corrected_path):
        raise FileNotFoundError(f"Corrected file not found: {corrected_path}")
    
    # Load and strip whitespace from column names
    orig = pd.read_csv(original_path)
    orig.columns = orig.columns.str.strip()
    corr = pd.read_csv(corrected_path)
    corr.columns = corr.columns.str.strip()
    return orig, corr


def compare_token_pair(orig_df, corr_df, token_pair='token0&1'):
    """Compare results for a specific token pair."""
    orig_tp = orig_df[orig_df['token_pair'] == token_pair].sort_values('layer')
    corr_tp = corr_df[corr_df['token_pair'] == token_pair].sort_values('layer')
    
    if len(orig_tp) == 0 or len(corr_tp) == 0:
        print(f"Warning: No data found for {token_pair}")
        return None
    
    print("=" * 80)
    print(f"RoPE Correction Results - {token_pair}")
    print("=" * 80)
    print(f"\n{'Layer':>5} | {'Original':>8} | {'Corrected':>9} | {'Δ':>10} | {'Δ%':>9} | {'Status':>8}")
    print("-" * 80)
    
    improvements = []
    for _, row_o in orig_tp.iterrows():
        layer = row_o['layer']
        row_c = corr_tp[corr_tp['layer'] == layer]
        
        if len(row_c) == 0:
            continue
            
        row_c = row_c.iloc[0]
        improvement = row_c['mean'] - row_o['mean']
        pct_change = (improvement / row_o['mean']) * 100
        improvements.append(improvement)
        
        # Status indicator
        if improvement > 0.05:
            status = "✓✓"
        elif improvement > 0.01:
            status = "✓"
        elif improvement > -0.01:
            status = "≈"
        else:
            status = "✗"
        
        print(f"{int(layer):5d} | {row_o['mean']:8.4f} | {row_c['mean']:9.4f} | "
              f"{improvement:+10.4f} | {pct_change:+8.2f}% | {status:>8}")
    
    print("-" * 80)
    avg_orig = orig_tp['mean'].mean()
    avg_corr = corr_tp['mean'].mean()
    avg_improvement = avg_corr - avg_orig
    avg_pct = (avg_improvement / avg_orig) * 100
    
    print(f"Average:  {avg_orig:8.4f} | {avg_corr:9.4f} | {avg_improvement:+10.4f} | {avg_pct:+8.2f}%")
    print("=" * 80)
    print()
    
    return {
        'token_pair': token_pair,
        'avg_original': avg_orig,
        'avg_corrected': avg_corr,
        'avg_improvement': avg_improvement,
        'avg_pct_change': avg_pct,
        'improvements': improvements
    }


def summary_statistics(orig_df, corr_df):
    """Print summary statistics across all token pairs."""
    token_pairs = sorted(orig_df['token_pair'].unique())
    
    print("\n" + "=" * 80)
    print("Summary Statistics - All Token Pairs")
    print("=" * 80)
    print(f"\n{'Token Pair':>12} | {'Avg Original':>13} | {'Avg Corrected':>14} | "
          f"{'Improvement':>12} | {'% Change':>10}")
    print("-" * 80)
    
    results = []
    for tp in token_pairs:
        orig_tp = orig_df[orig_df['token_pair'] == tp]
        corr_tp = corr_df[corr_df['token_pair'] == tp]
        
        if len(orig_tp) == 0 or len(corr_tp) == 0:
            continue
        
        avg_orig = orig_tp['mean'].mean()
        avg_corr = corr_tp['mean'].mean()
        improvement = avg_corr - avg_orig
        pct_change = (improvement / avg_orig) * 100
        
        results.append({
            'token_pair': tp,
            'avg_original': avg_orig,
            'avg_corrected': avg_corr,
            'improvement': improvement,
            'pct_change': pct_change
        })
        
        print(f"{tp:>12} | {avg_orig:13.4f} | {avg_corr:14.4f} | "
              f"{improvement:+12.4f} | {pct_change:+9.2f}%")
    
    print("-" * 80)
    
    # Overall statistics
    if results:
        overall_orig = sum(r['avg_original'] for r in results) / len(results)
        overall_corr = sum(r['avg_corrected'] for r in results) / len(results)
        overall_imp = sum(r['improvement'] for r in results) / len(results)
        overall_pct = (overall_imp / overall_orig) * 100
        
        print(f"{'Overall':>12} | {overall_orig:13.4f} | {overall_corr:14.4f} | "
              f"{overall_imp:+12.4f} | {overall_pct:+9.2f}%")
    
    print("=" * 80)
    print()


def analyze_by_layer(orig_df, corr_df):
    """Analyze improvement by layer range."""
    print("\n" + "=" * 80)
    print("Analysis by Layer Range")
    print("=" * 80)
    
    # Group by layer ranges
    ranges = [
        (1, 5, "Early layers (1-5)"),
        (6, 15, "Middle layers (6-15)"),
        (16, 32, "Late layers (16-32)")
    ]
    
    for start, end, label in ranges:
        orig_range = orig_df[(orig_df['layer'] >= start) & (orig_df['layer'] <= end)]
        corr_range = corr_df[(corr_df['layer'] >= start) & (corr_df['layer'] <= end)]
        
        if len(orig_range) == 0 or len(corr_range) == 0:
            continue
        
        avg_orig = orig_range['mean'].mean()
        avg_corr = corr_range['mean'].mean()
        improvement = avg_corr - avg_orig
        pct_change = (improvement / avg_orig) * 100
        
        print(f"{label:>25}: {avg_orig:.4f} → {avg_corr:.4f} "
              f"({improvement:+.4f}, {pct_change:+.2f}%)")
    
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare query similarity results")
    parser.add_argument('--original', type=str, 
                       default='logs/query_sim_qhead_summary.csv',
                       help='Path to original CSV file')
    parser.add_argument('--corrected', type=str,
                       default='logs/query_sim_rope_corrected.csv',
                       help='Path to corrected CSV file')
    parser.add_argument('--token-pair', type=str, default=None,
                       help='Specific token pair to analyze (e.g., token0&1)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        orig_df, corr_df = load_data(args.original, args.corrected)
        
        print(f"\nLoaded data:")
        print(f"  Original:  {len(orig_df)} rows from {args.original}")
        print(f"  Corrected: {len(corr_df)} rows from {args.corrected}")
        print()
        
        # Analyze specific token pair or all
        if args.token_pair:
            compare_token_pair(orig_df, corr_df, args.token_pair)
        else:
            # Show all token pairs
            token_pairs = sorted(orig_df['token_pair'].unique())
            for tp in token_pairs[:3]:  # Show first 3 in detail
                compare_token_pair(orig_df, corr_df, tp)
        
        # Summary statistics
        summary_statistics(orig_df, corr_df)
        
        # Layer range analysis
        analyze_by_layer(orig_df, corr_df)
        
        print("\n✓ Analysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
