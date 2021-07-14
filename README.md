Papers read<br>
Record some papers that I have read ! 

## Survey papers
- [Fraud detection: A systematic literature review of graph-based anomaly detection approaches](https://www.researchgate.net/publication/340691343_Fraud_detection_A_systematic_literature_review_of_graph-based_anomaly_detection_approaches) **[Decision Support Systems 2020]**
- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf) **[IEEE 2020]**
- [Graph Neural Networks in Recommender Systems: A Survey](https://arxiv.org/abs/2011.02260) **[2020]**


## Recommender System
#### [Denoising Implicit Feedback for Recommendation](https://arxiv.org/abs/2006.04153) **[SIGIR 2020]**
- **Authors**: Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, Tat-Seng Chua
- **Dataset**: Adressa, Amazon-book, Yelp
- **Note**: New training strategy (Truncated Loss and Reweighted Loss)
- **Task**: Recommendation (Implicit)
- [Code](https://github.com/WenjieWWJ/DenoisingRec)

#### [Global Context Enhanced Graph Neural Networks for Session-based Recommendation](https://dl.acm.org/doi/pdf/10.1145/3397271.3401142) **[SIGIR 2020]**
- **Authors**: Ziyang Wang, Wei Wei, Gao Cong, Xiao-Li Li, Xian-Ling Mao, Minghui Qiu
- **Dataset**: Diginetica, Tmall, Nowplaying
- **Note**: Global graph and session graph
- **Task**: Session-based Recommendation
- [Code](https://github.com/johnny12150/GCE-GNN)

#### [LightGCN](https://arxiv.org/pdf/2002.02126.pdf) **[SIGIR 2020]**
- **Authors**: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang
- **Dataset**: Gowalla, Yelp2018 31, Amazon-Book 
- **Note**: Two essential components — light graph convolution and layer combination
- **Task**: Recommendation (Implicit)
- [Code](https://github.com/gusye1234/LightGCN-PyTorch)

#### [Dynamic Graph Collaborative Filtering](https://arxiv.org/pdf/2101.02844.pdf) **[SIGIR 2020]**
- **Authors**: Xiaohan Li, Mengqi Zhang∗, Shu Wu‡, Zheng Liu, Liang Wang, Philip S.Yu
- **Dataset** : Reddit, Wikipedia, LastFM
- **Task** : Recommendation
- [Code](https://github.com/CRIPAC-DIG/DGCF)

#### [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://arxiv.org/pdf/1908.01207.pdf) **[KDD 2019]**
- **Authors** : Srijan Kumar, Xikun Zhang, Jure Leskovec
- **Dataset** : Reddit, Wikipedia, LastFM, MOOC
- **Task** : Recommendation (predicting future interactions and state change prediction)
- [Code](https://github.com/srijankr/jodie)

#### [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/pdf/1905.08108.pdf) **[AAAI 2019]**
- **Authors** : Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, Tieniu Tan
- **Dataset** : Yoochoose 1/64, Yoochoose 1/4, Diginetica
- **Task** : Session-based Recommendation
- [Code](https://github.com/CRIPAC-DIG/SR-GNN)


#### [Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf) **[SIGIR 2019]**
- **Authors** : Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, Tat-Seng Chua
- **Dataset** : Gowalla, Yelp2018, Amazon-Book
- **Task** : Recommendation (Implicit)
- [Code](https://github.com/huangtinglin/NGCF-PyTorch)

#### [Neural Collaborative Filtering](https://dl.acm.org/doi/10.1145/3038912.3052569) **[WWW 2017]**
- **Authors** : Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua
- **Dataset** : MovieLens, Pinterest
- **Task** : Recommendation (Implicit)
- [Code](https://github.com/hexiangnan/neural_collaborative_filtering)

## Static graph
#### [GRAPH ATTENTION NETWORKS](https://research.fb.com/wp-content/uploads/2018/03/graph-attention-networks.pdf) **[ICLR 2018]**
- **Research unit** : University of Cambridge, Centre de Visio per Computador, Facebook AI Research, Montreal Institute for Learning Algorithms
- **Dataset** : Cora, Citeseer, Pubmed, PPI(Inductive)
- **Task** : Node classification
- [Code](https://github.com/PetarV-/GAT)

#### [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) **[ICLR 2017]**
- **Authors** : Thomas N. Kipf, Max Welling
- **Dataset** : Cora, Citeseer, Pubmed, NELL
- **Task** : Node classification
- [Code](https://github.com/tkipf/gcn)

#### [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf) **[KDD 2014]**
- **Research unit** : Stony Brook University
- **Dataset** : BlogCatalog, Flickr, YouTube
- **Task** : Node classification
- [Code](https://github.com/phanein/deepwalk)


## Dynamic graph
#### [INDUCTIVE REPRESENTATION LEARNING ON TEMPORAL GRAPHS](https://arxiv.org/pdf/2002.07962.pdf) **[ICLR 2020]**
- **Research unit** : Walmart Labs
- **Dataset** : Reddit, Wikipedia, Industrial
- **Task** :  Node classification and link prediction task with transductive and inductive tasks 
- [Code](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)

#### [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/pdf/1902.10191.pdf) **[AAAI 2020]**
- **Research unit** : MIT-IBM Watson AI Lab, IBM Research, MIT CSAIL
- **Dataset** : SBM, BC-OTC, BC-Alpha, UCI, AS, Reddit, Elliptic
- **Task** :  Link prediction, Edge classification, Node classification 
- [Code](https://github.com/IBM/EvolveGCN)

#### [dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning](https://arxiv.org/pdf/1809.02657.pdf) **[Knowledge-Based Systems 2019]**
- **Research unit** :  University of Southern California, University of California-Irvine, Siemens Corporate Technology
- **Dataset** : SBM(Stochastic Block Model), Hep-th, AS
- **Task** : Link prediction
- [Code](https://github.com/palash1992/DynamicGEM)

#### [DynGEM: Deep Embedding Method for Dynamic Graphs](https://arxiv.org/pdf/1805.11273.pdf) **[CoRR 2018]**
- **Research unit** :  University of Southern California
- **Dataset** : SYN(synthetic dataset), HEP-TH, AS, ENRON
- **Task** :  Graph visualization, Graph reconstruction, Link prediction
and Anomaly detection
- **Note** : 
  - Construct dynamic graph embedding via autoencoder approach
  - Only avaliable in **growing graph**, ie, Node increasing over time

#### [DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks](http://yhwu.me/publications/dysat_wsdm20.pdf) **[WSDM 2020]**
- **Research unit** :  University of Illinois at Urbana-Champaign, Visa Research
- **Dataset** : Enron, UCI, Yelp, ML-10M
- **Task** :   Link prediction experiments on two graph types: communication networks and bipartite rating networks
- [Code](https://github.com/aravindsankar28/DySAT)

#### [Continuous-Time Dynamic Network Embeddings](http://ryanrossi.com/pubs/nguyen-et-al-WWW18-BigNet.pdf) **[WWW 2018]**
- **Research unit** :  Worcester Polytechnic Institute, Adobe Research, Intel Labs
- **Dataset** : ia-contact, ia-hypertext09, ia-enron-employees, ia-radoslaw-email, ia-email-eu, fb-forum, soc-bitcoinA, soc-wiki-elec 
- **Task** :   Temporal Link Prediction

## Emotion Recognition
#### [Compact Graph Architecture for Speech Emotion Recognition](https://arxiv.org/abs/2105.12907) **[ ICASSP 2021]**
- **Authors** : A. Shirian, T. Guha
- **Dataset** : IEMOCAP, MELK, DailyDialog, Emory NLP
- **Task** : Classification
- [Code](https://github.com/AmirSh15/Compact_SER)


#### [Directed Acyclic Graph Network for Conversational Emotion Recognition](https://arxiv.org/abs/2105.12907) **[ACL-IJCNLP 2021]**
- **Authors** : Weizhou Shen, Siyue Wu, Yunyi Yang, Xiaojun Quan
- **Dataset** : IEMOCAP,  MSP-IMPRO
- **Task** : (Graph) classification
- [Code](https://github.com/shenwzh3/DAG-ERC)


## Anomaly detection

#### [Few-shot Network Anomaly Detection via Cross-network Meta-learning](https://arxiv.org/pdf/2102.11165v1.pdf) **[WWW 2021]**
- **Research unit** : Arizona State University, University of Illinois at Urbana-Champaign,
- **Dataset** : Yelp, PubMed, Reddit
- **Task** : Detecting anomalies nodes 

#### [Anomaly detection in dynamic attributed networks](https://www.researchgate.net/publication/342399167_Anomaly_detection_in_dynamic_attributed_network) **[Neural Computing and Applications 2021]**
- **Research unit** : South China Normal University, Cloud and Smart Industries Group(Tencent), Ant Financial Services Group, University of Chinese Academy of Sciences, Rutgers University
- **Dataset** : Synthetic, Amazon, DBLP
- **Task** : Detecting anomalies nodes in dynamic attributed graphs

#### [ResGCN: Attention-based Deep Residual Modeling for Anomaly Detection on Attributed Networks](https://arxiv.org/pdf/2009.14738.pdf) **[CoRR 2020]**
- **Dataset** : Amazon, Enron, BlogCatalog, Flickr, ACM
- **Task** : Detecting anomalies nodes 

#### [Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs](https://arxiv.org/pdf/2005.07427.pdf) **[Preprint 2020]**
- **Research unit** :  Washington State University, NEC Laboratories America, Rice University
- **Dataset** : UCI, Digg, Email, Bitcoin-Alpha, Bitcoin-otc, Topology
- **Task** : Anomaly link prediction

#### [A Semi-supervised Graph Attentive Network for Financial Fraud Detection](https://arxiv.org/pdf/2003.01171.pdf) **[ICDM 2019]**
- **Research unit** : Ant Financial Services Group, Tsinghua University,China
- **Dataset** : ALIPAY
- **Task** : User Default Prediction,  User Attribute Prediction

#### [Anomaly Detection with Graph Convolutional Networks for Insider Threat and Fraud Detection](http://169.237.7.61/pubs/PID6150935_GCN.pdf) **[IEEE MILCOM 2019]**
- **Research unit** :  Chinese Academy of Sciences, University of California,  University of Texas at San Antonio,San Antonio
- **Dataset** :  Data set provided by CMU CERT consists of 1000 users in a simulated network and their activities from 01/02/2010 to 05/16/2011
- **Task** : Detecting anomalies nodes 

#### [TitAnt: Online Real-time Transaction Fraud Detection in Ant Financial](https://arxiv.org/pdf/1906.07407.pdf) **[VLDB 2019]**
- **Research unit** : Ant Financial Services Group
- **Task** : Detecting anomalies transactions

## Other
#### [GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media](https://www.aclweb.org/anthology/2020.acl-main.48.pdf) **[ACL 2020]**
- **Research unit** :  National Cheng Kung University
- **Dataset** : Twitter15, Twitter16
- **Task** :  Fake news prediction
- **Note** : 
  - Graph, GRU, CNN, Co-attention
- [Code](https://github.com/l852888/GCAN)
