#from .Criteo import Criteo
#from .iPinYou import iPinYou
from .tencent_ads_pcvr_contest_pre import TencentAdsPcvrContestPreData

def as_dataset(data_name, **kwargs):
    data_name = data_name.lower()
    return {
        #'criteo': Criteo(**kwargs),
        #'ipinyou': iPinYou(**kwargs),
        'tencent-ads-pcvr-contest-pre': TencentAdsPcvrContestPreData(**kwargs),
    }[data_name]
