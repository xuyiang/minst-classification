# coding:utf8

import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
import gc
import os

import models
from config import opt
from data.dataset import Minst
from utils.visualize import Visualizer


def train(**kwargs):
    """
    训练
    :param kwargs:
    :return:
    """
    try:
        print("Starting training...")
        opt.parse(kwargs)
        vis = Visualizer(opt.env)

        # Set device and seed for reproducibility
        device = t.device('cuda' if opt.use_gpu and t.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        t.manual_seed(opt.seed)
        if device.type == 'cuda':
            t.cuda.manual_seed(opt.seed)

        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)

        print("Loading model...")
        # model
        model = getattr(models, opt.model)()
        model.to(device)
        model.train()

        if opt.load_model_path:
            print(f"Loading model from {opt.load_model_path}")
            model.load(opt.load_model_path)

        print("Loading data...")
        # data
        train_data = Minst(data_root=opt.train_image_path, label_root=opt.train_label_path, train=True)
        val_data = Minst(data_root=opt.train_image_path, label_root=opt.train_label_path, train=False)

        print("Creating data loaders...")
        train_dataloader = DataLoader(train_data, opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.num_workers,
                                    pin_memory=True if device.type == 'cuda' else False)
        val_dataloader = DataLoader(val_data, opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers,
                                  pin_memory=True if device.type == 'cuda' else False)

        print("Setting up optimizer...")
        # 目标函数和优化器
        criterion = t.nn.CrossEntropyLoss()
        lr = opt.lr
        optimizer = t.optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=opt.weight_decay)

        # 统计指标，平滑处理之后的损失，还有混淆矩阵
        loss_meter = meter.AverageValueMeter()
        confusion_matrix = meter.ConfusionMeter(10)
        previous_loss = float('inf')

        print("Starting training loop...")
        for epoch in range(opt.max_epoch):
            print(f"\nEpoch {epoch+1}/{opt.max_epoch}")
            loss_meter.reset()
            confusion_matrix.reset()

            # Training loop
            model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Training")
            for ii, (data, target) in enumerate(progress_bar):
                # Move data to device
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # Forward pass
                optimizer.zero_grad()
                score = model(data)
                loss = criterion(score, target)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update metrics
                loss_meter.add(loss.item())
                confusion_matrix.add(score.detach(), target.detach())

                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss_meter.value()[0]:.4f}")

                if ii % opt.print_freq == opt.print_freq - 1:
                    vis.plot('loss', loss_meter.value()[0])

                # Clear some memory
                del data, target, score, loss
                if device.type == 'cuda':
                    t.cuda.empty_cache()

            # Save model
            print("Saving model...")
            model.save()

            # Validation
            print("Running validation...")
            val_cm, val_accuracy = val(model, val_dataloader, device)
            vis.plot('val_accuracy', val_accuracy)
            vis.log(f"epoch:{epoch}, lr:{lr}, loss:{loss_meter.value()[0]:.4f}, val_accuracy:{val_accuracy:.2f}%")

            # Learning rate adjustment
            if loss_meter.value()[0] > previous_loss:
                lr = lr * opt.lr_decay
                print(f"Reducing learning rate to {lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            previous_loss = loss_meter.value()[0]

            # Garbage collection
            gc.collect()
            if device.type == 'cuda':
                t.cuda.empty_cache()

    except Exception as e:
        print(f"Error during training: {e}")
        raise


def val(model, dataloader, device):
    """
    计算模型在验证集上的准确率等信息，用于辅助训练
    :param model:
    :param dataloader:
    :param device:
    :return:
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(10)

    with t.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for ii, (input, target) in enumerate(progress_bar):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            score = model(input)
            confusion_matrix.add(score.detach(), target.detach())

            # Clear some memory
            del input, target, score
            if device.type == 'cuda':
                t.cuda.empty_cache()

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value.diagonal().sum()) / (cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    """
    测试
    :param kwargs:
    :return:
    """
    try:
        print("Starting testing...")
        opt.parse(kwargs)

        # Set device
        device = t.device('cuda' if opt.use_gpu and t.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Configure model
        print("Loading model...")
        model = getattr(models, opt.model)().eval()
        if opt.load_model_path:
            model.load(opt.load_model_path)
        model.to(device)

        # Data
        print("Loading test data...")
        test_data = Minst(data_root=opt.test_image_path, test=True)
        test_dataloader = DataLoader(test_data, batch_size=opt.batch_size,
                                   shuffle=False,
                                   num_workers=opt.num_workers,
                                   pin_memory=True if device.type == 'cuda' else False)

        results = []
        print("Running inference...")
        with t.no_grad():
            progress_bar = tqdm(test_dataloader, desc="Testing")
            for ii, (input, path) in enumerate(progress_bar):
                input = input.to(device, non_blocking=True)
                score = model(input)
                probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()

                batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]
                results += batch_results

                # Clear some memory
                del input, score, probability
                if device.type == 'cuda':
                    t.cuda.empty_cache()

        print(f"Testing complete. Processed {len(results)} samples.")
        return results

    except Exception as e:
        print(f"Error during testing: {e}")
        raise


def help():
    """
    打印辅助信息
    :return: 
    """
    pass


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
