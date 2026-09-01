from .models.seqconvattn import SeqConvAttnModel
from .models.malconv import MalConvModel
from .models.rtf_cannie import RTFCannieModel
from .models.rtf_bert import RTFBertModel
from .models.binary_malconv import BinaryMalConv
from .models.imcfn import IMCFNModel
from .models.multiview_cnn import MultiViewCNNModel
from .models.m_attn_health import MAttnHealthModel
from .models.m_attn_health_multi import MAttnHealthModel as MAttnHealthMultiModel

MODEL_REGISTRY = {
    "seqconvattn": SeqConvAttnModel,
    "malconv": MalConvModel,
    "rtf_cannie": RTFCannieModel,
    "rtf_bert": RTFBertModel,
    "binary_malconv": BinaryMalConv,
    "imcfn": IMCFNModel,
    "multiview_cnn": MultiViewCNNModel,
    "m_attn_health": MAttnHealthModel,
    "m_attn_health_multi": MAttnHealthMultiModel,
}