# job_offers_classifier/__init__.py
from .job_offers_classfier_old import (
    ChineseLinearJobOffersClassifier,
    ChineseTransformerJobOffersClassifier,
    create_chinese_job_classifier,
    get_recommended_chinese_models
)

__all__ = [
    'ChineseLinearJobOffersClassifier',
    'ChineseTransformerJobOffersClassifier', 
    'create_chinese_job_classifier',
    'get_recommended_chinese_models'
]