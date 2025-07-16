"""
手語辨識完整流程執行腳本 v1
使用方式:
1. python run_pipeline_v1.py --step preprocess  # 只執行資料預處理
2. python run_pipeline_v1.py --step train       # 只執行模型訓練
3. python run_pipeline_v1.py --step test        # 只執行模型測試
4. python run_pipeline_v1.py --step all         # 執行完整流程
"""

import argparse
import sys
import os
from datetime import datetime

def run_preprocessing():
    """執行資料預處理"""
    print("開始執行資料預處理...")
    try:
        from data_preprocessing_v1 import SignLanguageDataProcessor
        processor = SignLanguageDataProcessor()
        processor.run_preprocessing()
        print("✅ 資料預處理完成!")
        return True
    except Exception as e:
        print(f"❌ 資料預處理失敗: {e}")
        return False

def run_training():
    """執行模型訓練"""
    print("開始執行模型訓練...")
    try:
        from train_model_v1 import SignLanguageTrainer
        trainer = SignLanguageTrainer()
        trainer.run_training(
            epochs=50,          # 可根據需要調整
            batch_size=16,
            learning_rate=0.001,
            sequence_length=20
        )
        print("✅ 模型訓練完成!")
        return True
    except Exception as e:
        print(f"❌ 模型訓練失敗: {e}")
        return False

def run_testing():
    """執行模型測試"""
    print("開始執行模型測試...")
    try:
        from test_model_v1 import SignLanguageTester
        tester = SignLanguageTester()
        tester.run_testing(
            model_path=None,
            sequence_length=20,
            batch_size=32
        )
        print("✅ 模型測試完成!")
        return True
    except Exception as e:
        print(f"❌ 模型測試失敗: {e}")
        return False

def check_requirements():
    """檢查必要的套件是否已安裝"""
    required_packages = [
        'pandas', 'numpy', 'torch', 'sklearn', 
        'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下必要套件:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n請先安裝缺少的套件:")
        print("pip install torch pandas numpy scikit-learn matplotlib seaborn joblib")
        return False
    
    print("✅ 所有必要套件都已安裝")
    return True

def check_data_folder():
    """檢查資料夾是否存在"""
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        print(f"❌ 找不到資料夾: {dataset_folder}")
        return False
    
    # 檢查是否有CSV文件
    import glob
    csv_files = glob.glob(os.path.join(dataset_folder, "sign*.csv"))
    if not csv_files:
        print(f"❌ 在 {dataset_folder} 中找不到以sign開頭的CSV文件")
        return False
    
    print(f"✅ 找到 {len(csv_files)} 個CSV文件")
    return True

def main():
    parser = argparse.ArgumentParser(description='手語辨識完整流程執行腳本 v1')
    parser.add_argument('--step', choices=['preprocess', 'train', 'test', 'all'], 
                       default='all', help='要執行的步驟')
    parser.add_argument('--check-only', action='store_true', 
                       help='只檢查環境，不執行任何步驟')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("手語辨識流程執行器 v1")
    print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 環境檢查
    print("\n🔍 環境檢查...")
    env_ok = True
    
    if not check_requirements():
        env_ok = False
    
    if not check_data_folder():
        env_ok = False
    
    if not env_ok:
        print("\n❌ 環境檢查失敗，請先解決上述問題。")
        return 1
    
    if args.check_only:
        print("\n✅ 環境檢查通過，可以開始執行流程。")
        return 0
    
    # 執行指定步驟
    success = True
    
    if args.step in ['preprocess', 'all']:
        print("\n" + "="*50)
        if not run_preprocessing():
            success = False
            if args.step == 'all':
                print("由於預處理失敗，停止執行後續步驟。")
                return 1
    
    if args.step in ['train', 'all']:
        print("\n" + "="*50)
        if not run_training():
            success = False
            if args.step == 'all':
                print("由於訓練失敗，停止執行後續步驟。")
                return 1
    
    if args.step in ['test', 'all']:
        print("\n" + "="*50)
        if not run_testing():
            success = False
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 所有步驟執行完成!")
        if args.step == 'all':
            print("完整的手語辨識模型開發流程已完成。")
            print("您可以在以下位置找到結果:")
            print("- 處理後的資料集: v1/processed_data/")
            print("- 訓練好的模型: v1/models/")
            print("- 測試結果: v1/results/")
    else:
        print("❌ 執行過程中出現錯誤，請檢查上述錯誤訊息。")
        return 1
    
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
