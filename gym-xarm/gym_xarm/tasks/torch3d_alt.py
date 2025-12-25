
import torch
import numpy as np
from termcolor import cprint

def farthest_point_sampling_torch(points, num_points, device='cuda'):
    """
    Farthest Point Sampling (FPS) using PyTorch.
    
    Args:
        points (torch.Tensor): Point cloud of shape (N, D) or (N, 3).
        num_points (int): Number of points to sample.
        device (str): Device to run on ('cuda' or 'cpu').
        
    Returns:
        torch.Tensor: Sampled points of shape (num_points, D).
        torch.Tensor: Indices of sampled points.
    """
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).to(device)
    
    N, D = points.shape
    if N < num_points:
        # Pad with zeros if fewer points than requested
        zeros = torch.zeros((num_points - N, D), device=device)
        return torch.cat([points, zeros], dim=0), torch.arange(num_points, device=device)
        
    centroids = torch.zeros(num_points, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)
    
    batch_indices = torch.arange(N, device=device)
    
    for i in range(num_points):
        centroids[i] = farthest
        centroid = points[farthest, :3].view(1, 3)
        dist = torch.sum((points[:, :3] - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance)
        
    return points[centroids], centroids

def point_cloud_sampling(point_cloud: np.ndarray, num_points: int, method: str = 'uniform', device: str = None):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb
    """
    if num_points == 'all': # use all points
        return point_cloud
    if point_cloud.shape[0] <= num_points:
        # pad with zeros
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], 6))], axis=0)
        return point_cloud
    
    if method == 'uniform':
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == 'fps':
        # fast point cloud sampling using pytorch
        if not device:
            if torch.backends.mps.is_available():
                device = 'mps'
            if torch.cuda.is_available():
                device = 'cuda'
        else:
            device = 'cpu'
            
        # Convert to torch for sampling
        sampled_points, _ = farthest_point_sampling_torch(point_cloud, num_points, device=device)
        # Convert back to numpy
        point_cloud = sampled_points.cpu().numpy()
    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")

    return point_cloud

