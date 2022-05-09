def show_height_map(height_map, visualize_height_map=True, save_height_map_path=None, save_for_material=False, color_map='PuBuGn'):
    if visualize_height_map or save_height_map_path is not None:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import imageio

        if isinstance(height_map, str):
            height_map_filename = height_map.split('/')[-1]
            height_map = np.array(imageio.imread(height_map))
        elif save_height_map_path is not None:
            height_map_filename = save_height_map_path.split('/')[-1]
        else:
            height_map_filename = ''

        mpl.rcParams['toolbar'] = 'None'

        height_map = np.clip(height_map, -1, 1)
        height_map = 65535 * (height_map + 1) / 2

        plt.imshow(height_map, aspect='auto', interpolation='none', cmap=color_map)
        plt.xlim(0, np.shape(height_map)[1] - 1)
        plt.ylim(np.shape(height_map)[0] - 1, 0)

        fig = plt.gcf()
        fig.canvas.set_window_title(height_map_filename)
        plt.tight_layout()

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        if save_height_map_path is not None:
            import png
            writer = png.Writer(width=height_map.shape[1], height=height_map.shape[0], bitdepth=16, greyscale=True)
            with open(save_height_map_path, 'wb') as f:
                writer.write(f, height_map.astype(int).tolist())

            if save_for_material:
                plt.savefig(save_height_map_path)

        if visualize_height_map:
            plt.show()
