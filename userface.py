# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 17:35
# @Author  : Leesure
# @File    : userface.py
# @Software: PyCharm
import hanlp
import pickle
from preprocess_drcd import build_adjacency_matrix, pos_token_to_ids
from inference import Generator
import config
print("[*] Load Hanlp model")
tokenizer = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
print("[*] Load Edge Vocab")
edge_vocab = pickle.load(open('/home1/liyue/Lee_NQG/dataset/preprocess/DRCD/edge_vocab.pt', 'rb'))
print("[*] Local Vocab")
pos_vocab = pickle.load(open('/home1/liyue/Lee_NQG/dataset/preprocess/DRCD/pos_vocab.pt', 'rb'))
print("[*] Embed Vector")
embed_vector = pickle.load(open(config.embed_vector_file, 'rb'))


def token_sentence(sent: str, ans: str, ans_start: int):
    S_hanlp = tokenizer(sent, tasks=['dep', 'pos'])
    S_tokens = S_hanlp['tok/fine']
    S_dep = S_hanlp['dep']
    S_pos = S_hanlp['pos/ctb']
    S_pos_ids = pos_token_to_ids(S_pos, pos_vocab)
    ans_end = ans_start + len(ans)
    mask_ans = build_adjacency_matrix(S_dep, S_tokens, answer_start, ans_end, edge_vocab)
    print(S_tokens)
    return [{
        'S': S_tokens,
        'Q': None,
        'ans': ans,
        'A': mask_ans,
        'S_pos': S_pos_ids
    }]


if __name__ == '__main__':
    print('input sentence')
    sentence = '线性表更关注的是单个元素的操作，比如查找一个元素，插入或删除一个元素'
    answer = '单个元素'
    answer_start = 8
    # sentence = '新北市总人口中客家人口约占14.1％，虽非全台湾最高，但总人口数为全台湾第二多，仅次于桃园市；新北市客家人分布较为分散，主要区域包括新庄、三重、三峡、新店安坑、汐止、泰山、五股、土城、深坑等地；另外东北角石门、金山一带亦有少数客家人迁徙至当地定居。新北市的原住民人口占总人口比例1.29%，以泰雅族为主，主要分布在乌来区；乌来区同时也是新北市唯一的直辖市山地原住民区。另外，新北市新住民人口约有九万七千多人，为台湾新住民最多的城市，若把新住民子女及新移工纳入，总人口数超过20万人。另外中和区的华新街，为早期南洋华侨迁居来此地定居逐渐形成的聚落，尤其以缅甸人为主，每年「泼水节」活动为商圈重要节庆；永和区的中兴街则为早期韩国华侨的聚集所在，又有『韩国街』之称。'
    # sentence = '2010年引进的广州快速公交运输系统，属世界第二大快速公交系统，日常载客量可达100万人次，高峰时期每小时单向客流高达26900人次，仅次于波哥大的快速交通系统，平均每10秒钟就有一辆巴士，每辆巴士单向行驶350小时。包括桥梁在内的站台是世界最长的州快速公交运输系统站台，长达260米。目前广州市区的计程车和公共汽车主要使用液化石油气作燃料，部分公共汽车更使用油电、气电混合动力技术。2012年底开始投放液化天然气燃料的公共汽车，2014年6月开始投放液化天然气插电式混合动力公共汽车，以取代液化石油气公共汽车。2007年1月16日，广州市政府全面禁止在市区内驾驶摩托车。违反禁令的机动车将会予以没收。广州市交通局声称禁令的施行，使得交通拥挤问题和车祸大幅减少。广州白云国际机场位于白云区与花都区交界，2004年8月5日正式投入运营，属中国交通情况第二繁忙的机场。该机场取代了原先位于市中心的无法满足日益增长航空需求的旧机场。目前机场有三条飞机跑道，成为国内第三个拥有三跑道的民航机场。比邻近的香港国际机场第三跑道预计的2023年落成早8年。'
    # answer :广州的快速公交运输系统每多久就会有一辆巴士？
    # print('input answer')
    # answer = '桃园市'
    # print('answer_start')
    # answer_start = 43
    print('[*] Generate question')
    example = token_sentence(sentence, answer, answer_start)
    checkpoint = '/home1/liyue/Lee_NQG/checkpoint/model_26/loss_0.3617.pt'
    generator = Generator(checkpoint, None, print_question=True, example=example)
    generator.decode()
