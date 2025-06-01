# 在Python命令行中测试
from job_offers_classifier.job_offers_classfier_old import ChineseTransformerJobOffersClassifier

# 创建一个测试实例
test_hierarchy = {'1234': {'parents': ['1', '12', '123'], 'label': '1234', 'level': 4}}

classifier = ChineseTransformerJobOffersClassifier(
    model_dir='./test',
    hierarchy=test_hierarchy,
    use_hierarchical_loss=True
)

# 检查属性是否存在
print("use_hierarchical_loss:", hasattr(classifier, 'use_hierarchical_loss'))
print("use_multitask_learning:", hasattr(classifier, 'use_multitask_learning'))
print("属性值:", classifier.use_hierarchical_loss, classifier.use_multitask_learning)