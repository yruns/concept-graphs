"""分段结果可视化工具"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_signals(debug_info, segments, output_path=None, figsize=(16, 10)):
    n_frames = debug_info['n_frames']
    frames = np.arange(n_frames)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    motion = debug_info.get('motion_signals', {})
    fused = debug_info.get('fused_signals', {})
    change_points = debug_info.get('change_points', [])
    
    ax = axes[0]
    if motion:
        ax.plot(frames, motion.get('motion_intensity', []), label='Motion', lw=2)
    ax.set_ylabel('Motion')
    ax.legend()
    ax.set_title('Temporal Scene Segmentation')
    
    ax = axes[1]
    if fused:
        ax.plot(frames, fused.get('fused_signal', []), label='Fused', lw=2, color='purple')
    for cp in change_points:
        ax.axvline(x=cp, color='red', ls='--', alpha=0.7)
    ax.set_ylabel('Fused')
    ax.legend()
    
    ax = axes[2]
    n_segs = max(len(segments), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_segs))
    for i, seg in enumerate(segments):
        sd = seg.to_dict() if hasattr(seg, 'to_dict') else seg
        ax.axvspan(sd['start_frame'], sd['end_frame'], alpha=0.5, color=colors[i])
        mid = (sd['start_frame'] + sd['end_frame']) / 2
        ax.text(mid, 0.5, 'R%d' % sd['region_id'], ha='center', va='center', fontweight='bold')
    ax.set_ylabel('Segments')
    ax.set_xlabel('Frame')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print('Saved to: %s' % output_path)
    return fig


def generate_summary(segments, debug_info, output_path):
    lines = ['=' * 50, 'Segmentation Report', '=' * 50]
    lines.append('Frames: %d' % debug_info['n_frames'])
    lines.append('Segments: %d' % len(segments))
    lines.append('Change points: %s' % str(debug_info.get('change_points', [])))
    lines.append('')
    
    for seg in segments:
        sd = seg.to_dict() if hasattr(seg, 'to_dict') else seg
        n_obj = len(sd.get('object_indices', []))
        lines.append('Region %d: frames %d-%d (%d frames), %d objects' % (
            sd['region_id'], sd['start_frame'], sd['end_frame'], sd['n_frames'], n_obj))
    
    report = '\n'.join(lines)
    Path(output_path).write_text(report)
    print(report)
    return report
