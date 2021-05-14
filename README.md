# Graph-related-papers-
Record some graph related papers that I have read ! 

## Survey papers
- [Fraud detection: A systematic literature review of graph-based anomaly detection approaches](https://www.researchgate.net/publication/340691343_Fraud_detection_A_systematic_literature_review_of_graph-based_anomaly_detection_approaches)**[Decision Support Systems 2020]**
- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf)**[IEEE 2020]**
- [Graph Neural Networks in Recommender Systems: A Survey](https://arxiv.org/abs/2011.02260) **[2020]**


## Recommender System
#### [Dynamic Graph Collaborative Filtering](https://arxiv.org/pdf/2101.02844.pdf)**[SIGIR 2020]**
- **Research unit** : University of Illinois at Chicago, School of Artificial Intelligence, University of Chinese Academy of Sciences, Institute of Automation, Chinese Academy of Sciences
- **Dataset** : Reddit, Wikipedia, LastFM
- **Task** :  Link prediction
- [Code](https://github.com/CRIPAC-DIG/DGCF)

## Static graph
#### [GRAPH ATTENTION NETWORKS](https://research.fb.com/wp-content/uploads/2018/03/graph-attention-networks.pdf) **[ICLR 2018]**
- **Research unit** : University of Cambridge, Centre de Visio per Computador, Facebook AI Research, Montreal Institute for Learning Algorithms
- **Dataset** : Cora, Citeseer, Pubmed, PPI(Inductive)
- **Task** : Node classification
- [Code](https://github.com/PetarV-/GAT)

#### [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) **[ICLR 2017]
- **Research unit** : University of Amsterdam, Canadian Institute for Advanced Research (CIFAR)
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
