"""
時序長度處理演示腳本
展示不同長度動作如何被處理成固定長度的序列
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_sequence_processing():
    """視覺化演示序列處理過程"""
    
    # 模擬不同長度的手語動作
    action_data = {
        'hello': 20,    # 20幀
        'thanks': 25,   # 25幀 (剛好等於sequence_length)
        'goodbye': 35,  # 35幀
        'sorry': 15     # 15幀
    }
    
    sequence_length = 25  # 目標序列長度
    
    print("=" * 60)
    print("手語動作時序長度處理演示")
    print("=" * 60)
    print(f"目標序列長度: {sequence_length} 幀")
    print("-" * 60)
    
    results = {}
    
    for action, length in action_data.items():
        print(f"\n處理動作: {action} ({length} 幀)")
        
        if length >= sequence_length:
            # 使用滑動窗口
            num_sequences = length - sequence_length + 1
            print(f"  策略: 滑動窗口")
            print(f"  生成序列數: {num_sequences}")
            print(f"  序列範圍:")
            for i in range(min(num_sequences, 3)):  # 只顯示前3個
                start = i + 1
                end = i + sequence_length
                print(f"    序列{i+1}: 幀{start}-{end}")
            if num_sequences > 3:
                print(f"    ... (共{num_sequences}個序列)")
                
            results[action] = {
                'strategy': '滑動窗口',
                'sequences': num_sequences,
                'padding': 0
            }
            
        else:
            # 使用填充策略
            padding_needed = sequence_length - length
            print(f"  策略: 填充")
            print(f"  原始長度: {length} 幀")
            print(f"  填充長度: {padding_needed} 幀")
            print(f"  填充方式: 重複最後一幀")
            print(f"  最終序列: 幀1-{length} + {padding_needed}×幀{length}")
            
            results[action] = {
                'strategy': '填充',
                'sequences': 1,
                'padding': padding_needed
            }
    
    print("\n" + "=" * 60)
    print("處理結果摘要")
    print("=" * 60)
    
    total_sequences = 0
    for action, result in results.items():
        total_sequences += result['sequences']
        strategy = result['strategy']
        sequences = result['sequences']
        
        if strategy == '填充':
            padding = result['padding']
            print(f"{action:<10}: {strategy:<8} → {sequences} 序列 (填充{padding}幀)")
        else:
            print(f"{action:<10}: {strategy:<8} → {sequences} 序列")
    
    print("-" * 60)
    print(f"總訓練序列數: {total_sequences}")
    
    # 分析樣本分布
    print("\n📊 樣本分布分析:")
    for action, result in results.items():
        percentage = (result['sequences'] / total_sequences) * 100
        print(f"{action:<10}: {result['sequences']:>2} 序列 ({percentage:>5.1f}%)")
    
    return results

def demonstrate_sliding_window():
    """演示滑動窗口的具體過程"""
    print("\n" + "="*50)
    print("滑動窗口詳細演示 - 'goodbye' (35幀)")
    print("="*50)
    
    total_frames = 35
    sequence_length = 25
    
    print(f"原始動作: 35幀 [1, 2, 3, ..., 35]")
    print(f"序列長度: {sequence_length}幀")
    print(f"可生成序列數: {total_frames - sequence_length + 1} = {35-25+1} = 11個")
    print("\n序列內容:")
    
    for i in range(total_frames - sequence_length + 1):
        start = i + 1
        end = i + sequence_length
        if i < 5 or i >= 8:  # 顯示前5個和最後3個
            print(f"序列{i+1:>2}: 幀[{start:>2}-{end:>2}]")
        elif i == 5:
            print("    ...")
    
def demonstrate_padding():
    """演示填充的具體過程"""
    print("\n" + "="*50)
    print("填充策略詳細演示 - 'sorry' (15幀)")
    print("="*50)
    
    original_frames = 15
    sequence_length = 25
    padding_needed = sequence_length - original_frames
    
    print(f"原始動作: {original_frames}幀")
    print(f"目標長度: {sequence_length}幀")
    print(f"需要填充: {padding_needed}幀")
    print()
    
    # 模擬原始資料
    print("原始序列:")
    original = [f"幀{i+1}" for i in range(original_frames)]
    print(f"[{', '.join(original)}]")
    
    print("\n填充後序列:")
    padded = original + [f"幀{original_frames}"] * padding_needed
    print(f"[{', '.join(padded)}]")
    print(f"  {'|--原始15幀--|':>40} {'|--填充10幀--|':>25}")

if __name__ == "__main__":
    # 執行演示
    results = visualize_sequence_processing()
    demonstrate_sliding_window()
    demonstrate_padding()
    
    print("\n" + "="*60)
    print("結論:")
    print("1. 長動作使用滑動窗口 → 增加訓練樣本")
    print("2. 短動作使用填充策略 → 維持序列長度一致")
    print("3. 需要注意樣本分布平衡問題")
    print("4. 可以考慮動態調整sequence_length以優化效果")
    print("="*60)
