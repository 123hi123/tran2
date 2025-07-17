#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑動窗口機制說明工具
解釋為什麼231幀可以創建212個序列
"""

def explain_sliding_window(total_frames, sequence_length):
    """解釋滑動窗口機制"""
    print(f"🔍 滑動窗口機制解析")
    print("=" * 60)
    print(f"總幀數: {total_frames}")
    print(f"序列長度: {sequence_length}")
    print()
    
    # 計算序列數量
    num_sequences = total_frames - sequence_length + 1
    print(f"📊 序列數量計算:")
    print(f"num_sequences = total_frames - sequence_length + 1")
    print(f"              = {total_frames} - {sequence_length} + 1")
    print(f"              = {num_sequences}")
    print()
    
    # 視覺化前幾個序列
    print("🎬 前10個序列的範圍:")
    print("-" * 40)
    for i in range(min(10, num_sequences)):
        start = i
        end = i + sequence_length - 1
        print(f"序列 {i+1:2d}: 幀 [{start:3d} - {end:3d}]")
    
    if num_sequences > 10:
        print("...")
        # 顯示最後幾個序列
        for i in range(max(num_sequences-3, 10), num_sequences):
            start = i
            end = i + sequence_length - 1
            print(f"序列 {i+1:2d}: 幀 [{start:3d} - {end:3d}]")
    
    print()
    print("🎯 關鍵觀察:")
    print(f"- 第1個序列: 幀 [0 - {sequence_length-1}]")
    print(f"- 第2個序列: 幀 [1 - {sequence_length}]")
    print(f"- 第3個序列: 幀 [2 - {sequence_length+1}]")
    print(f"- ...")
    print(f"- 第{num_sequences}個序列: 幀 [{num_sequences-1} - {total_frames-1}]")
    
    # 檢查邊界
    last_start = num_sequences - 1
    last_end = last_start + sequence_length - 1
    print()
    print("✅ 邊界檢查:")
    print(f"最後一個序列的結束位置: {last_end}")
    print(f"總幀數的最後一幀索引: {total_frames-1}")
    print(f"是否匹配: {'✅' if last_end == total_frames-1 else '❌'}")
    
    return num_sequences

def visualize_overlap(total_frames=10, sequence_length=4):
    """小例子可視化重疊"""
    print(f"\n📈 小例子可視化 (總共{total_frames}幀，序列長度{sequence_length})")
    print("=" * 60)
    
    num_sequences = total_frames - sequence_length + 1
    
    # 顯示所有幀
    frame_display = "幀編號: " + " ".join([f"{i:2d}" for i in range(total_frames)])
    print(frame_display)
    print()
    
    # 顯示每個序列
    for seq_idx in range(num_sequences):
        start = seq_idx
        end = seq_idx + sequence_length
        
        # 創建視覺化字串
        visual = ["  " for _ in range(total_frames)]
        for i in range(start, end):
            visual[i] = f"{i:2d}"
        
        print(f"序列{seq_idx+1}: " + " ".join(visual) + f"  <-- 幀[{start}-{end-1}]")
    
    print()
    print("💡 重要概念:")
    print("- 每個序列都是連續的幀")
    print("- 相鄰序列之間只差1幀 (滑動步長=1)")
    print("- 這樣可以最大化訓練數據的利用")

def check_sequence_quality():
    """檢查序列品質"""
    print(f"\n🎯 為什麼要用滑動窗口？")
    print("=" * 60)
    
    print("✅ 優點:")
    print("1. 最大化數據利用 - 不浪費任何幀")
    print("2. 增加訓練樣本數 - 231幀 → 212個序列")
    print("3. 捕捉時間變化 - 每個序列都包含連續動作")
    print("4. 提高模型泛化 - 看到更多動作變化")
    
    print("\n⚠️  注意事項:")
    print("1. 序列間高度重疊 - 可能過擬合")
    print("2. 計算量增加 - 更多序列要處理")
    print("3. 需要足夠長的原始動作")
    
    print(f"\n🔬 你的案例分析:")
    print("- 'RELAY-OPERATOR' 有231幀")
    print("- 這是一個相對長的手語動作")
    print("- 滑動窗口可以捕捉動作的不同階段")
    print("- 212個序列可以訓練模型識別這個手語的各種變化")

def alternative_approaches():
    """其他可能的方法"""
    print(f"\n🛠️  其他序列分割方法:")
    print("=" * 60)
    
    total_frames = 231
    sequence_length = 20
    
    print("1️⃣ 滑動窗口 (目前使用):")
    num_sliding = total_frames - sequence_length + 1
    print(f"   序列數: {num_sliding}")
    print(f"   重疊度: 高 (每次移動1幀)")
    
    print("\n2️⃣ 無重疊分割:")
    num_no_overlap = total_frames // sequence_length
    print(f"   序列數: {num_no_overlap}")
    print(f"   重疊度: 無")
    print(f"   浪費幀數: {total_frames % sequence_length}")
    
    print("\n3️⃣ 固定步長 (例如步長=5):")
    stride = 5
    num_stride = (total_frames - sequence_length) // stride + 1
    print(f"   序列數: {num_stride}")
    print(f"   重疊度: 中等")
    
    print(f"\n💭 為什麼選擇滑動窗口？")
    print("- 手語是連續動作，需要捕捉細微變化")
    print("- 更多訓練樣本有助於模型學習")
    print("- 序列間的小幅重疊實際上是有益的")

def main():
    # 你的實際案例
    total_frames = 231
    sequence_length = 20  # 從程式碼推測
    
    num_sequences = explain_sliding_window(total_frames, sequence_length)
    
    # 小例子幫助理解
    visualize_overlap(10, 4)
    
    # 檢查設計合理性
    check_sequence_quality()
    
    # 其他方法比較
    alternative_approaches()
    
    print(f"\n🏁 結論:")
    print("=" * 60)
    print(f"✅ 代碼設計正確！")
    print(f"✅ 231幀創建212個序列是合理的")
    print(f"✅ 滑動窗口是手語辨識的標準做法")
    print(f"✅ 可以有效訓練模型識別連續動作")

if __name__ == "__main__":
    main()
