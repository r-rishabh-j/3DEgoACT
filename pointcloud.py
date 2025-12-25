from flask import Flask, render_template_string
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.cm as cm
from termcolor import cprint
import os


def visualize_pointcloud(pointcloud, color:tuple=None):
    vis = Visualizer()
    vis.visualize_pointcloud(pointcloud, color=color)

def visualize_pointclouds_video(pointclouds, path):
    vis = Visualizer()
    vis.visualize_pointcloud_video(pointclouds, 10, path)

class Visualizer:
    def __init__(self):
        self.app = Flask(__name__)
        self.pointclouds = []
        
    def _generate_trace(self, pointcloud, color:tuple=None, size=5, opacity=0.7):
        x_coords = pointcloud[:, 0]
        y_coords = pointcloud[:, 1]
        z_coords = pointcloud[:, 2]

        if pointcloud.shape[1] == 3:
            if color is None:
                # design a colorful point cloud based on 3d coordinates
                # Normalize coordinates to range [0, 1]
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
                try:
                    # Use normalized coordinates as RGB values
                    colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
                except: # maybe meet NaN error
                    # use simple cyan color
                    colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]
            else:    
                colors = ['rgb({},{},{})'.format(color[0], color[1], color[2]) for _ in range(len(x_coords))]
        else:
            colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in pointcloud[:, 3:6]]

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=size,
                opacity=opacity,
                color=colors
            )
        )


    def colorize(self, pointcloud):
        if pointcloud.shape[1] == 3:

            # design a colorful point cloud based on 3d coordinates
            # Normalize coordinates to range [0, 1]
            min_coords = pointcloud.min(axis=0)
            max_coords = pointcloud.max(axis=0)
            normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
            try:
                # Use normalized coordinates as RGB values
                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
            except: # maybe meet NaN error
                # use simple cyan color
                x_coords = pointcloud[:, 0]
                colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]

        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]
        return colors
    

    def visualize_pointcloud(self, pointcloud, color:tuple=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        div = pio.to_html(fig, full_html=False)

        @self.app.route('/')
        def index():
            return render_template_string('''<div>{{ div|safe }}</div>''', div=div)
        
        self.app.run(debug=True, use_reloader=False)
    
    def visualize_pointcloud_video(self, pointclouds, fps=10, save_path="pointcloud_video.html"):
        """
        Visualizes a sequence of point clouds as an animation.
        
        Args:
            pointclouds: List of numpy arrays, where each array is a point cloud frame.
            fps: Frames per second for the animation.
            save_path: Path to save the HTML animation.
        """
        
        frames = []
        
        # Determine global bounds for consistent axis scaling
        print('####', pointclouds)
        all_points = np.concatenate(pointclouds, axis=0)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

        for k, pc in enumerate(pointclouds):
            trace = self._generate_trace(pc, size=6, opacity=1.0)
            frames.append(go.Frame(data=[trace], name=str(k)))

        # Initial trace
        initial_trace = self._generate_trace(pointclouds[0], size=6, opacity=1.0)
        
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(range=[x_min, x_max], showbackground=False, showgrid=True, showline=True, linecolor='grey', gridcolor='grey', zeroline=False),
                yaxis=dict(range=[y_min, y_max], showbackground=False, showgrid=True, showline=True, linecolor='grey', gridcolor='grey', zeroline=False),
                zaxis=dict(range=[z_min, z_max], showbackground=False, showgrid=True, showline=True, linecolor='grey', gridcolor='grey', zeroline=False),
                bgcolor='white'
            ),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 1000/fps, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 0, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [{
                    "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": str(k),
                    "method": "animate"
                } for k in range(len(pointclouds))]
            }]
        )

        fig = go.Figure(data=[initial_trace], layout=layout, frames=frames)
        
        # Save to HTML
        div = pio.to_html(fig, full_html=False, auto_play=False)
        
        # Create a simple Flask app to serve the video
        @self.app.route('/video')
        def video():
            return render_template_string('''<div>{{ div|safe }}</div>''', div=div)
            
        print(f"Video visualization ready. Run the app and visit /video. Also saving to {save_path}")
        
        # Also save to file directly
        with open(save_path, 'w') as f:
            f.write(pio.to_html(fig, full_html=True, auto_play=False))

        # self.app.run(debug=True, use_reloader=False)

    def visualize_pointcloud_and_save(self, pointcloud, color:tuple=None, save_path=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        # save
        fig.write_image(save_path, width=800, height=600)
        

    def save_visualization_to_file(self, pointcloud, file_path, color:tuple=None):
        # visualize pointcloud and save as html
        trace = self._generate_trace(pointcloud, color=color)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_html = pio.to_html(go.Figure(data=[trace], layout=layout), full_html=True)

        with open(file_path, 'w') as file:
            file.write(fig_html)
        print(f"Visualization saved to {file_path}")
    