import numpy as np

class WeightMap:
    def __init__(self, size_x=400, size_y=400, initial_weight=1.0, update_weight=0.5, decay_rate=0.1, radius= 30):
        self.size_x = size_x
        self.size_y = size_y
        self.map = np.full((size_x, size_y), initial_weight)
        self.decay_rate = decay_rate
        self.update_weight = update_weight
        self.radius = radius
        self.probabilities = np.zeros((size_x, size_y))

    def update(self, position):
        # Increase the weight at the current position and surrounding cells within radius
        for i in range(max(0, position[0]-self.radius), min(self.size_x, position[0]+self.radius+1)):
            for j in range(max(0, position[1]-self.radius), min(self.size_y, position[1]+self.radius+1)):
                distance = np.sqrt((i - position[0])**2 + (j - position[1])**2)
                if distance <= self.radius:
                    weight_increase = (1 - distance/self.radius) * self.update_weight
                    self.map[i, j] -= weight_increase

        # Decrease the weight at the current position
        self.map += self.decay_rate

        # Ensure the weight does not go below zero
        # self.map[position] = max(self.map[position], 0)

    def get_next_target(self):
        # Normalize the weights to make them probabilities
        self.map = self.map + np.abs(np.min(self.map))
        probabilities = self.map / np.sum(self.map)
        print(np.min(self.map))
        print(np.max(self.map))

        # Flatten the probabilities to 1D
        probabilities_1d = probabilities.flatten()
        print(np.sum(probabilities_1d))

        # Choose a random index based on the probabilities
        target_index_1d = np.random.choice(len(probabilities_1d), p=probabilities_1d)

        # Convert the 1D index back to 2D coordinates
        target_x, target_y = np.unravel_index(target_index_1d, (self.size_x, self.size_y))

        self.probabilities = probabilities

        return (target_x, target_y)

def interpolate_movement(start_position, end_position):
    sx, sy = start_position[0], start_position[1]
    ex, ey = end_position[0], end_position[1]

    # Calculate the distance between the start and end positions
    distance = np.sqrt((sx - ex)**2 + (sy - ey)**2)

    # Calculate the number of steps to take
    num_steps = int(distance / 15)

    if num_steps == 0:
        return [start_position]

    # Calculate the step size
    step_size = 1.0 / num_steps

    # Calculate the x and y step sizes
    x_step = (ex - sx) * step_size
    y_step = (ey - sy) * step_size

    # Create a list of positions to move to
    positions = []
    for i in range(num_steps):
        positions.append((sx + i * x_step, sy + i * y_step))

    return positions


weight_map = WeightMap(400, 400)
current_position = (0, 0)
weight_map.update(current_position)

weightmap_list = [np.array(weight_map.probabilities)]
for i in range(150):
    next_position = weight_map.get_next_target()

    # Interpolate the movement between the current position and the next position
    positions = interpolate_movement(current_position, next_position)
    for i in positions:
        weight_map.update((int(i[0]), int(i[1])))

        weight_map.update(next_position)

    map = np.array(weight_map.probabilities)

    weightmap_list.append(map)

    current_position = next_position

weightmap_list = np.array(weightmap_list)

# visualizing the weight map with matplotlib as a heatmap as a gif
import matplotlib.pyplot as plt
import imageio
import os

def save_weight_maps_as_gif(weight_maps, filename):
    # Create a list to hold the filenames of the individual frames
    filenames = []

    for i, weight_map in enumerate(weight_maps):
        # Create a plot of the weight map
        plt.imshow(weight_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        # Save the plot as an image
        frame_filename = f"frame_{i}.png"
        plt.savefig(frame_filename)
        filenames.append(frame_filename)
        plt.close()  # Close the plot to free up memory

    # Combine the frames into a gif
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Delete the individual frames
    for filename in filenames:
        os.remove(filename)

save_weight_maps_as_gif(weightmap_list, 'weight_maps_test.gif')