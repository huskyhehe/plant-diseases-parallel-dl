        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        
    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)




def plot_by_plant(data):
    barwidth = 0.2   # inch per bar
    spacing = 4      # spacing between subplots in units of barwidth
    figx = 5         # figure width in inch
    left = 4         # left margin in units of bar width
    right = 2        # right margin in units of bar width
    tc = len(data)   # "total_categories", holds how many charts to create
    max_values = []  # holds the maximum number of bars to create
    
    for category, entries in data.items():
        max_values.append(len(entries))
    max_values = np.array(max_values)
    # total figure height:
    figy = ((np.sum(max_values) + tc) + (tc + 1) * spacing) * barwidth  # inch

    fig = plt.figure(figsize=(figx, figy))
    ax = None
    color_cycle = plt.colormaps.get_cmap('viridis').colors
    for index, (category, entries) in enumerate(data.items()):
        entry_names_values = sorted(entries.items(), key=lambda x: x[0] == 'healthy')
        entry_names, values = zip(*entry_names_values)

        if not entry_names:
            continue  # do not create empty charts

        y_ticks = range(1, len(entry_names) + 1)
        coord = [left * barwidth / figx,
                 1 - barwidth * ((index + 1) * spacing + np.sum(max_values[:index + 1]) + index + 1) / figy,
                 1 - (left + right) * barwidth / figx,
                 (max_values[index] + 1) * barwidth / figy]

        ax = fig.add_axes(coord, sharex=ax)
        ax.barh(y_ticks, values, color=color_cycle[index * (len(color_cycle) // len(data))])
        ax.set_ylim(0, max_values[index] + 1)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(entry_names)
        ax.invert_yaxis()
        ax.set_title(category, loc="left")
    
    plt.show()

plot_by_plant(plant_dict)
