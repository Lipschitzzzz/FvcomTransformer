import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import time
import os
import numpy as np
import fvcomtransformer

def train_zero_epoch_ddp(
        node_data_dir,
        triangle_data_dir,
        num_epochs,
        checkpoint_name_out,
        total_timesteps=144 * 7,
        steps_per_file=144,
        input_steps=1,
        pred_step=1,
        early_stop_patience=25
):
    start_time = time.time()
    best_loss = float('inf')
    early_stop_cnt = 0
    train_loss_dataset = []
    train_loss_dataset1 = []
    train_loss_dataset2 = []
    val_loss_dataset = []
    val_loss_dataset1 = []
    val_loss_dataset2 = []
    lr_history = []
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    model = fvcomtransformer.FvcomModel(
        node=60882, triangle=115443, node_var=13,
        triangle_var=17, embed_dim=256,
        n_heads=8, num_transformer_layers=6, dropout=0.1
    ).to(device)

    full_dataset = fvcomtransformer.FvcomDataset(
        node_data_dir=node_data_dir,
        triangle_data_dir=triangle_data_dir,
        total_timesteps=total_timesteps,
        steps_per_file=steps_per_file,
        input_steps=input_steps,
        pred_step=pred_step
    )

    model = DDP(model, device_ids=[local_rank])

    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler,
        num_workers=2, pin_memory=True
    )

    steps_per_epoch = len(train_loader)
    total_epochs = 0 + num_epochs
    total_steps = steps_per_epoch * total_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # criterion = elementtransformer.WeightedMAEMSELoss().to(device)
    # criterion = torch.nn.MSELoss()

    # huber = torch.nn.HuberLoss(delta=0.1, reduction='mean').to(device)
    # mae = torch.nn.L1Loss(reduction='mean').to(device)
    # criterion = torch.nn.L1Loss(reduction='mean').to(device)
    criterion = torch.nn.L1Loss(reduction='mean').to(device)
    # criterion = torch.nn.SmoothL1Loss(beta=1.0)

    # W_tri = 1.0
    # W_node = 0.3

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = torch.tensor(0.0, device=device)
        train_loss_node = torch.tensor(0.0, device=device)
        train_loss_triangle = torch.tensor(0.0, device=device)
        train_count = torch.tensor(0.0, device=device)
        iter = 0
        for (node_x, tri_x), (node_y, tri_y) in train_loader:
            node_x, tri_x = node_x.to(device), tri_x.to(device)
            node_y, tri_y = node_y.to(device), tri_y.to(device)

            node_pred, tri_pred = model(node_x, tri_x)
            if local_rank == 0 and iter % 20 == 0:
                print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "input node:      ", node_x.shape)
                print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "input triangle:  ", tri_x.shape)
                print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "target node:     ", node_y.shape)
                print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "target triangle: ", tri_y.shape)
                print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "pred node:       ", node_pred.shape)
                print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "pred triangle:   ", tri_pred.shape)

            loss_node = criterion(node_pred, node_y)
            loss_triangle = criterion(tri_pred, tri_y)
            loss = loss_node + loss_triangle
            train_loss_sum += loss.detach()
            train_loss_node += loss_node.detach()
            train_loss_triangle += loss_triangle.detach()
            train_count += 1
            iter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss_node, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss_triangle, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count, op=dist.ReduceOp.SUM)

        train_loss = (train_loss_sum / train_count).item()
        train_loss_1 = (train_loss_node / train_count).item()
        train_loss_2 = (train_loss_triangle / train_count).item()



        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        val_loss_node = torch.tensor(0.0, device=device)
        val_loss_triangle = torch.tensor(0.0, device=device)
        val_count = torch.tensor(0.0, device=device)
        iter = 0
        with torch.no_grad():
            for (node_x, tri_x), (node_y, tri_y) in val_loader:
                node_x, tri_x = node_x.to(device), tri_x.to(device)
                node_y, tri_y = node_y.to(device), tri_y.to(device)

                node_pred, tri_pred = model(node_x, tri_x)
                
                loss_node = criterion(node_pred, node_y)
                loss_triangle = criterion(tri_pred, tri_y)
                loss = loss_node + loss_triangle
                if local_rank == 0 and iter % 20 == 0:
                    print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "input node:      ", node_x.shape)
                    print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "input triangle:  ", tri_x.shape)
                    print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "target node:     ", node_y.shape)
                    print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "target triangle: ", tri_y.shape)
                    print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "pred node:       ", node_pred.shape)
                    print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "pred triangle:   ", tri_pred.shape)

                val_loss_sum += loss.detach()
                val_loss_node += loss_node.detach()
                val_loss_triangle += loss_triangle.detach()

                val_count += 1
                iter += 1

        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_node, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_triangle, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        val_loss = (val_loss_sum / val_count).item()
        val_loss_1 = (val_loss_node / val_count).item()
        val_loss_2 = (val_loss_triangle / val_count).item()

        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best: {best_loss:.6f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss Node: {train_loss_1:.6f}, Train Loss Triangle: {train_loss_2:.6f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss Node: {val_loss_1:.6f}, Val Loss Triangle: {val_loss_2:.6f}")
            train_loss_dataset.append(train_loss)
            train_loss_dataset1.append(train_loss_1)
            train_loss_dataset2.append(train_loss_2)
            val_loss_dataset.append(val_loss)
            val_loss_dataset1.append(val_loss_1)
            val_loss_dataset2.append(val_loss_2)
            lr_history.append(optimizer.param_groups[0]['lr'])
            
            np.save('train_loss.npy', train_loss_dataset)
            np.save('train_loss_node.npy', train_loss_dataset1)
            np.save('train_loss_triangle.npy', train_loss_dataset2)
            np.save('val_loss.npy', val_loss_dataset)
            np.save('val_loss_node.npy', val_loss_dataset1)
            np.save('val_loss_triangle.npy', val_loss_dataset2)
            np.save("lr.npy", lr_history)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                early_stop_cnt = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_loss_node': train_loss_1,
                    'train_loss_triangle': train_loss_2,
                    'val_loss': val_loss,
                    'val_loss_node': val_loss_1,
                    'val_loss_triangle': val_loss_2,
                }, checkpoint_name_out)

                print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}")
            else:
                early_stop_cnt += 1
                print(f"No improvement. Current val_loss: {val_loss:.6f}, Best so far: {best_loss:.6f}, Best epoch {best_epoch}")
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss Node: {train_loss_1:.6f}, Train Loss Triangle: {train_loss_2:.6f}")
                print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss Node: {val_loss_1:.6f}, Val Loss Triangle: {val_loss_2:.6f}")
            
        stop_flag = torch.tensor(0, device=device)
        if local_rank == 0 and early_stop_cnt > early_stop_patience:
            print("Early stopped.")
            stop_flag.fill_(1)
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

    total_time = time.time() - start_time
    if local_rank == 0:
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    print(world_size, "GPU found")
    assert world_size > 0, "No GPUs available"
    start_time = time.time()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime(start_time))
    train_zero_epoch_ddp(node_data_dir="dataset/node/data/",
                         triangle_data_dir="dataset/triangle/data/",
                         num_epochs=200,
                         checkpoint_name_out="checkpoints/" + timestamp_str+ "_ddp_training.pth",
                         total_timesteps=144*7,
                         input_steps=1,
                         pred_step=1)
    # train_from_pth_ddp(node_data_dir="dataset/node/data/",
    #                    tri_data_dir="dataset/triangle/data/",
    #                    num_epochs=100,
    #                    checkpoint_name_out="checkpoints/" + timestamp_str+ "_2A100.pth",
    #                    checkpoint_path='checkpoints/2025_12_11_13_59_2A100.pth',
    #                    total_timesteps=24*(30+31+30+31),
    #                    steps_per_file=24,
    #                    pred_step= 1,
    #                    )
if __name__ == "__main__":
    main()