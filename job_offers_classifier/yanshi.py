#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文职业分类完整演示流程
使用哈工大BERT模型进行ISCO职业编码分类
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# 导入你的分类器
from job_offers_classifier.job_offers_classfier_old import (
    create_chinese_job_classifier,
    ChineseTransformerJobOffersClassifier,
    get_recommended_chinese_models
)
from job_offers_classifier.job_offers_utils_old import create_hierarchy
from job_offers_classifier.load_save import save_as_text, load_texts

def create_demo_data():
    """创建演示数据"""
    print("🔄 创建演示数据...")
    
    # 1. 创建ISCO层次结构数据
    hierarchy_data = {
        'class': ['1', '11', '111', '1111', '1112', '2', '21', '211', '2111', '2112'],
        'name': [
            '经理人员', '首席执行官和高级官员', '立法机关成员和高级官员', 
            '立法机关成员', '高级政府官员',
            '专业技术人员', '科学和工程专业人员', '物理和地球科学专业人员',
            '物理学家和天文学家', '气象学家'
        ]
    }
    hierarchy_df = pd.DataFrame(hierarchy_data)
    
    # 2. 创建中文职业描述训练数据
    job_descriptions = [
        "负责公司整体战略规划，制定年度经营目标，管理高级管理团队",
        "制定公司发展战略，监督各部门运营，向董事会汇报工作",
        "参与国家立法工作，审议法律法案，代表人民行使权力",
        "组织人大会议，审查政府工作报告，监督法律实施",
        "制定政府政策，协调各部门工作，处理重大公共事务",
        "从事物理学研究，设计实验方案，分析实验数据，发表学术论文",
        "研究天体物理现象，使用天文望远镜观测，分析天体数据",
        "分析气象数据，制作天气预报，研究气候变化规律",
        "测量大气参数，建立气象模型，预测极端天气事件",
        "研究量子物理理论，进行粒子物理实验，开发新的物理模型"
    ]
    
    # 对应的ISCO编码
    job_labels = ['1111', '1111', '1111', '1111', '1112', '2111', '2111', '2112', '2112', '2111']
    
    # 3. 创建测试数据
    test_descriptions = [
        "担任公司CEO，负责企业管理和战略决策",
        "进行核物理研究，操作粒子加速器设备",
        "分析卫星气象图像，预报未来三天天气"
    ]
    test_expected = ['1111', '2111', '2112']  # 期望的预测结果
    
    return hierarchy_df, job_descriptions, job_labels, test_descriptions, test_expected

def save_demo_files(hierarchy_df, job_descriptions, job_labels, test_descriptions):
    """保存演示文件"""
    print("💾 保存演示文件...")
    
    # 创建演示目录
    demo_dir = Path("demo_chinese_job_classification")
    demo_dir.mkdir(exist_ok=True)
    
    # 保存层次结构
    hierarchy_path = demo_dir / "isco_hierarchy.csv"
    hierarchy_df.to_csv(hierarchy_path, index=False, encoding='utf-8')
    
    # 保存训练数据
    train_texts_path = demo_dir / "train_texts.txt"
    train_labels_path = demo_dir / "train_labels.txt"
    save_as_text(str(train_texts_path), job_descriptions)
    save_as_text(str(train_labels_path), job_labels)
    
    # 保存测试数据
    test_texts_path = demo_dir / "test_texts.txt"
    save_as_text(str(test_texts_path), test_descriptions)
    
    print(f"✅ 演示文件已保存到: {demo_dir}")
    return demo_dir, hierarchy_path, train_texts_path, train_labels_path, test_texts_path

def demonstrate_training():
    """演示完整的训练和测试流程"""
    print("🚀 开始中文职业分类演示")
    print("=" * 60)
    
    # 1. 创建演示数据
    hierarchy_df, job_descriptions, job_labels, test_descriptions, test_expected = create_demo_data()
    
    # 2. 保存演示文件
    demo_dir, hierarchy_path, train_texts_path, train_labels_path, test_texts_path = save_demo_files(
        hierarchy_df, job_descriptions, job_labels, test_descriptions
    )
    
    # 3. 显示推荐的中文模型
    print("\n📋 可用的中文BERT模型:")
    models = get_recommended_chinese_models()
    for key, info in models.items():
        status = "⭐" if info['recommended'] else "  "
        print(f"  {status} {key}: {info['model_name']}")
        print(f"     {info['description']}")
    
    # 4. 创建层次结构
    print(f"\n🏗️  构建ISCO层次结构...")
    hierarchy = create_hierarchy(hierarchy_df)
    print(f"✅ 层次结构包含 {len(hierarchy)} 个职业类别")
    
    # 5. 创建分类器
    print(f"\n🤖 创建中文BERT职业分类器...")
    model_dir = demo_dir / "chinese_bert_model"
    
    # 为了演示速度，使用较小的参数
    classifier = create_chinese_job_classifier(
        classifier_type='transformer',
        model_dir=str(model_dir),
        hierarchy=hierarchy,
        transformer_model='hfl/chinese-roberta-wwm-ext',  # 哈工大RoBERTa
        max_epochs=2,  # 演示用，实际应该更多
        batch_size=4,   # 演示用小batch
        learning_rate=2e-5,
        early_stopping=True,
        early_stopping_patience=1,
        verbose=True
    )
    
    print(f"✅ 分类器配置完成")
    print(f"   模型: hfl/chinese-roberta-wwm-ext (哈工大RoBERTa)")
    print(f"   训练样本: {len(job_descriptions)} 条")
    print(f"   职业类别: {len(set(job_labels))} 个")
    
    # 6. 训练模型
    print(f"\n🎯 开始训练...")
    try:
        classifier.fit(job_labels, job_descriptions)
        print("✅ 模型训练完成！")
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        print("💡 如果是GPU/CUDA问题，可以尝试使用CPU:")
        print("   添加参数: devices=1, accelerator='cpu'")
        return
    
    # 7. 测试预测
    print(f"\n🔮 测试预测...")
    print("测试数据:")
    for i, desc in enumerate(test_descriptions):
        print(f"  {i+1}. {desc[:30]}...")
    
    try:
        # 进行预测
        predictions_array, pred_mapping = classifier.predict(test_descriptions, format='array', top_k=3)
        predictions_df = classifier.predict(test_descriptions, format='dataframe', top_k=3)
        
        print(f"\n📊 预测结果:")
        print(predictions_df)
        
        # 8. 分析结果
        print(f"\n📈 详细分析:")
        for i, (desc, expected) in enumerate(zip(test_descriptions, test_expected)):
            predicted = predictions_df.iloc[i]['class_1']
            confidence = predictions_df.iloc[i]['prob_1']
            
            status = "✅" if predicted == expected else "❌"
            print(f"{status} 样本 {i+1}:")
            print(f"   描述: {desc[:40]}...")
            print(f"   预期: {expected}")
            print(f"   预测: {predicted} (置信度: {confidence:.3f})")
            
            # 显示top-3预测
            print(f"   Top-3: ", end="")
            for j in range(3):
                cls = predictions_df.iloc[i][f'class_{j+1}']
                prob = predictions_df.iloc[i][f'prob_{j+1}']
                print(f"{cls}({prob:.3f}) ", end="")
            print()
        
        # 9. 模型信息
        print(f"\n🔧 模型信息:")
        model_info = classifier.get_model_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ 预测出错: {e}")
        return
    
    # 10. 保存模型使用说明
    print(f"\n💡 使用说明:")
    print(f"模型已保存到: {model_dir}")
    print(f"加载使用:")
    print(f"""
    # 加载已训练的模型
    classifier = ChineseTransformerJobOffersClassifier()
    classifier.load('{model_dir}')
    
    # 预测新数据
    new_predictions = classifier.predict(['软件工程师开发移动应用程序'])
    """)
    
    print(f"\n🎉 演示完成！")

def quick_linear_demo():
    """快速线性模型演示（如果BERT太慢）"""
    print("⚡ 快速线性模型演示")
    print("=" * 40)
    
    # 创建数据
    hierarchy_df, job_descriptions, job_labels, test_descriptions, test_expected = create_demo_data()
    hierarchy = create_hierarchy(hierarchy_df)
    
    # 创建线性分类器（更快）
    from job_offers_classifier.job_offers_classfier_old import ChineseLinearJobOffersClassifier
    
    classifier = ChineseLinearJobOffersClassifier(
        model_dir="./demo_linear_model",
        hierarchy=hierarchy,
        verbose=True
    )
    
    print("🎯 训练线性模型...")
    classifier.fit(job_labels, job_descriptions)
    
    print("🔮 预测结果...")
    predictions_df = classifier.predict(test_descriptions, format='dataframe', top_k=3)
    print(predictions_df)
    
    print("✅ 线性模型演示完成！")

if __name__ == "__main__":
    print("中文职业分类系统演示")
    print("使用哈工大BERT模型")
    print("=" * 60)
    
    choice = input("选择演示模式:\n1. 完整BERT演示 (需要GPU，较慢但效果好)\n2. 快速线性演示 (CPU友好，较快)\n请输入 1 或 2: ")
    
    if choice == "1":
        demonstrate_training()
    elif choice == "2":
        quick_linear_demo()
    else:
        print("无效选择，运行完整演示...")
        demonstrate_training()