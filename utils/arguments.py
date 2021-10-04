import argparse
import sys


def parse_args():
    # dataset and method
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--method',                nargs='?',  default='tenet', help='gmf,transformer,gnn,tenet')
    parser.add_argument('--path',                  nargs='?',  default='./data/aotm/', help='Input data path.')
    parser.add_argument('--dataset',               nargs='?',  default='aotm', help='Choose a dataset.')
    parser.add_argument('--res_path',              nargs='?',  default='./saved_models/', help='result path for plots and best error values.')
    parser.add_argument('--res_folder',            nargs='?',  default='test', help='specific folder corresponding to different runs on different parameters.')
    parser.add_argument('--include_networks',      nargs='?',  default="['gnn', 'seq']", help='include given networks in the model.')
    #parser.add_argument('--include_networks',      nargs='?',  default="['transformer','gnn']",help='loss based on the given interactions.')

    # algo-parameters
    parser.add_argument('--num_epochs',            type=int,   default=1, help='Number of epochs.')
    parser.add_argument('--batch_size',            type=int,   default=2048, help='Batch size.')##2048
    parser.add_argument('--batch_size_seq',        type=int,   default=256, help='Seq batch size.') ##256
    parser.add_argument('--valid_batch_siz',       type=int,   default=32, help='Valid batch size.') ##seq, 32
    parser.add_argument('--lr',                    type=float, default=.001, help='Learning rate.')
    parser.add_argument('--optimizer',             nargs='?',  default='adam', help='adam')
    parser.add_argument('--loss',                  nargs='?',  default='ce', help='ce')
    parser.add_argument('--initializer',           nargs='?',  default='xavier', help='xavier')
    parser.add_argument('--stddev',                type=float, default=0.02, help='stddev for normal and [min,max] for uniform')
    parser.add_argument('--max_item_seq_length',   type=int,   default=200, help='number of rated items to keep.') #20 ## to cover all the items (20,50,200)
    parser.add_argument('--load_embedding_flag',   type=int,   default=0, help='0-->donot load embedding, 1-->load embedding for entities.')
    parser.add_argument('--at_k',                  type=int,   default=5, help='@k for recall, map and ndcg, etc.')
    parser.add_argument('--knn_k',                 type=int,   default=50, help='@k for knn.')
    parser.add_argument('--cosine',                nargs='?',  default='False', help='knn_graph cosine or not.')
    parser.add_argument('--embed_type',            nargs='?',  default='node2vec', help='Choose a dataset.')

    # hyper-parameters
    parser.add_argument('--num_factors',           type=int,   default=80, help='Embedding size.')
    parser.add_argument('--num_negatives',         type=int,   default=1, help='Negative instances in sampling.')
    parser.add_argument('--num_negatives_seq',     type=int,   default=2, help='Negative instances in sampling for seq (done in main itself).')
    parser.add_argument('--reg_w',                 type=float, default=0.0000, help="Regularization for weight vector.")
    parser.add_argument('--reg_b',                 type=float, default=0.000, help="Regularization for user and item bias embeddings.")
    parser.add_argument('--reg_lambda',            type=float, default=0.000, help="Regularization lambda for user and item embeddings.")
    parser.add_argument('--margin',                type=float, default=2.0, help='margin value for TripletMarginLoss.')
    parser.add_argument('--keep_prob',             type=float, default=0.5, help='droupout keep probability in layers.') #0.7##

    # gnn
    parser.add_argument('--num_layers',            type=int,   default=2, help='Number of hidden layers.') # feature in testing ##not completed
    parser.add_argument('--hid_units',             nargs='?',  default='[48,32]', help='hidden units of GAT')
    parser.add_argument('--gnn_keep_prob',         type=float, default=1.0, help='proj keep probability in projection weights layers for reviews.') #0.4
    parser.add_argument('--net_keep_prob',         type=float, default=1.0, help='proj keep probability in projection weights layers for reviews.')

    # multi-head
    parser.add_argument('--n_heads',               nargs='?',  default='[1]', help='number of heads of GAT')
    parser.add_argument('--d_k',                   type=int,   default=64, help='Number of hidden layers.') # feature in testing ##not completed

    # valid and test
    parser.add_argument('--dataset_avg_flag_zero', type=int,  default=0,  help='Dataset item embed zero (or) avg. zero --> 1, else avg')
    parser.add_argument('--epoch_mod',             type=int,  default=15,  help='epoch mod --> to display valid and test error.')
    parser.add_argument('--num_thread',            type=int,  default=16, help='number of threads.')
    parser.add_argument('--comment',               nargs='?', default='comment', help='comments about the current experimental iterations.')

    # new
    parser.add_argument('--store_embedding',       nargs='?', default='False', help='whether to store user-list-item embeddings for knn_graph.')
    parser.add_argument('--knn_graph',             nargs='?', default='True', help='knn_graph for tenet.')
    parser.add_argument('--user_adj_weights',      nargs='?', default='False', help='whether to use adjacency matrix weights for gnn.')
    parser.add_argument('--self_loop',             nargs='?', default='True', help='whether to use adjacency matrix weights for gnn.')

    # new
    parser.add_argument('--warm_start_gnn',         type=int,  default=100,  help='warm_start done on gnn part to give better embeddings to seq part.')
    parser.add_argument('--include_hgnn',           nargs='?', default='True', help='whether to include hgnn in gnn part of the network.')


    return parser.parse_args()
