from __future__ import division

from models_mod import*
from utils.utils_ori import *
from utils.datasets_multi import *
from utils.parse_config import *
from utils.logger import*
from test_kaist_multi import*

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3-kaist-multi.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/kaist_night_all.data", help="path to data config file")
parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
parser.add_argument("--weights_path", type=str, help="path to weights file/ can use darknet53.conv.74")
parser.add_argument("--class_path", type=str, default="data/kaist.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=12, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
opt = parser.parse_args()
# print(opt)
my_dataset = opt.data_config_path
# print('use'+ my_dataset)
#logger = Logger("logs") for tensorboard

freeze_backbone = 1
#vis = Visualizer('yolo v3')

cuda = torch.cuda.is_available()

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path_rgb = data_config["train_rgb"]
train_path_ir = data_config["train_ir"]
test_path_rgb = data_config["valid_rgb"]
test_path_ir = data_config["valid_ir"]
num_classes = int(data_config["classes"])
class_names = load_classes(data_config["names"])

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# check_point_path = 'checkpoints/'
# weights_files = os.listdir(check_point_path)
# weights_files = weights_files.sort()
# print(weights_files)
# weights_path_latest = check_point_path + weights_files[-1]
# print(weights_path_latest)



#initiate from checkpoint
if opt.weights_path:
    if opt.weights_path.endswith(".pth"):
        model.load_state_dict(torch.load(opt.pretrained_weights))
    else:
        model.load_darknet_weights(opt.pretrained_weights)


if cuda:
	device = torch.device("cuda")
	print("CUDA is ready")
else:
	device = torch.device("cpu")
    # model = model.cuda()

# Initiate model
model = Darknet(opt.model_config_path).to(device)
model.apply(weights_init_normal)    

# Get dataloader
train_dataset = ListDataset(list_path_rgb=train_path_rgb, list_path_ir=train_path_ir, augment=True, multiscale=True)
dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True, collate_fn=train_dataset.collate_fn
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(model.parameters())

losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
accumulated_batches = 4
best_mAP = 0.0


loss_data_file = open('loss data.txt','w+') #loss for each batch #start from 0 remember
test_data_file = open('test_data_map.txt','w+')

metrics =[
    "grid_size",
    "loss",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "cls",
    "cls_acc",
    "recall50",
    "recall75",
    "precision",
    "conf_obj",
    "conf_noobj",
]
print("start training")
train_start_time = time.time()
for epoch in range(opt.epochs):
    model.train()
    start_time = time.time() #time in second
    # #Freeze Darknet.Conv.74 for first layers
    # if freeze_backbone:
    #     if epoch < 20:
    #         for i, (name, p) in enumerate(model.named_parameters()):
    #             if int(name.split('.')[1]) < 75:  # if layer < 75
    #                 p.requires_grad = False
    #     elif epoch >= 20:
    #         for i, (name, p) in enumerate(model.named_parameters()):
    #             if int(name.split('.')[1]) < 75:  # if layer < 75
    #                 p.requires_grad = True           

    optimizer.zero_grad()   
    total_loss_for_save = 0.0       
    for batch_i, (imgs, targets) in enumerate(dataloader):
        
        batches_done = len(dataloader) * epoch + batch_i
        
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
       
        loss, outputs = model(imgs, targets)
        loss.backward()
        #optimizer.step()
                # accumulate gradient for x batches before optimizing
        if batches_done % opt.gradient_accumulations: #((batch_i + 1) % accumulated_batches == 0) or (batch_i == len(dataloader) - 1):
            optimizer.step()
            optimizer.zero_grad()
           
        # ------------------------------
        # Log progress
        # ------------------------------
        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
        
        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        #Log metrics at each YOLO layer
        for i, metric in enumerate(metrics):
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

            #Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"{name}_{j+1}", metric)]
            tensorboard_log += [("loss", loss.item())]
            #logger.list_of_scalars_summary(tensorboard_log, batches_done)
        log_str += AsciiTable(metric_table).table 
        log_str += f"\nTotal loss {loss.item()}"
        total_loss_for_save += loss.item()

        #Determine approximate time left for epoch
        epoch_batches_left = len(dataloader) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)

        model.seen += imgs.size(0)

        # losses_x += model.losses["x"]
        # losses_y += model.losses["y"]
        # losses_w += model.losses["w"]
        # losses_h += model.losses["h"]
        # losses_conf += model.losses["conf"]
        # losses_cls += model.losses["cls"]
        # losses_recall += model.losses["recall"]
        # losses_precision += model.losses["precision"]
        # batch_loss += loss.item() #total loss
        
 
        # print(
        #     "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
        #     % (
        #         epoch,
        #         opt.epochs,
        #         batch_i,
        #         len(dataloader),
        #         model.losses["x"],
        #         model.losses["y"],
        #         model.losses["w"],
        #         model.losses["h"],
        #         model.losses["conf"],
        #         model.losses["cls"],
        #         loss.item(),
        #         model.losses["recall"],
        #         model.losses["precision"],
        #     )
        # )
	

    #average of the loss
    # batches = len(dataloader) #batch number in 1 peoch
    # losses_x = losses_x / batches
    # losses_y = losses_y / batches
    # losses_w = losses_w / batches
    # losses_h = losses_h / batches
    # losses_conf = losses_conf / batches
    # losses_cls = losses_cls / batches
    # losses_recall = losses_recall / batches
    # losses_precision = losses_precision / batches
    # batch_loss = batch_loss / batches #total loss
   
    print("saving loss at epoch ", epoch)
    loss_data_file.write("%.5f\n" % (total_loss_for_save/len(dataloader)) ) #save average loss of all batches
    
    if epoch % opt.evaluation_interval == 0:
        print("\n---- Evaluating Model ----")
        #Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model, 
            path_rgb=test_path_rgb,
            path_ir=test_path_ir,
            iou_thres=0.5,
            conf_thres=0.8,
            nms_thres=0.4,
            img_size=opt.img_size,
            batch_size=8,
        )

        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]
        #logger.list_of_scalars_summary(evaluation_metrics, epoch)

    print(float(AP))
    print(ap_class)
    exit()
    #loss_data_file.write(loss_data)

    # Print Class APs and mAP
    ap_table = [["Index", "Class Name", "AP"]]
    for i, c in enumerate(ap_class):
    	ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    	#test_data_file.write("{0} {1} {2} {3} " % (precision[i], recall[i], AP[i], ap_class[i]) )
    print(AsciiTable(ap_table).table)
    mAP = AP.mean()
    print("--- mAP of all classes %.5f" % mAP)
    #test_data_file.write("%.5f\n" % mAP)

    if mAP > best_mAP:
    	best_mAP = mAP
    	model.save_darknet_weights("%s/kaist_best.weights" % (opt.checkpoint_dir))
    	print("New Best AP appear !!! %.5f" % best_mAP)
    	test_data_file.flush()

    time.sleep(2)
    if epoch % opt.checkpoint_interval == 0:
    	torch.save(model.state_dict(), "%s/%d.pth" % (opt.checkpoint_dir, epoch)) #save weight at every epoch

    # print("Average Precisions:")
    # for c, ap in average_precisions.items():
    #     print(f"+ Class '{c}' - AP: {ap}")
    #     test_data_file.write("%.5f "%ap)
    # mAP = np.mean(list(average_precisions.values()))
    # print(f"mAP: {mAP}")
    # test_data_file.write("%.5f\n"% mAP)
    
    # if(mAP > best_mAP):
    #     best_mAP = mAP
    #     model.save_weights("%s/kaist_best.weights" % (opt.checkpoint_dir))
    #     print("New Best AP appear !!! %f" % best_mAP)
    #     test_data_file.flush()
    print("Time passed for training: %.2f hours" % ((time.time()-train_start_time)/3600) )
    time.sleep(1)

loss_data_file.close()
test_data_file.close()

print("Time : %.2f hours"% ((time.time()-train_start_time) / 3600) ) #seconds divided by 3600