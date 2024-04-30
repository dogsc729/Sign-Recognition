import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    steps = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if line.startswith('Epoch'):
                parts = line.split(',')
                epoch_info = {}
                epoch_info['epoch'] = int(parts[0].split()[1])
                epoch_info['dev_loss'] = float(parts[1].split(':')[1])
                epoch_info['dev_accuracy'] = float(parts[2].split(':')[1].strip().split()[0])
                steps.append(epoch_info)
    return steps

def save_plot(data, filename, title, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(data['epoch'], data['value'], marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_data(steps, filename_prefix):
    dev_losses = {'epoch': [step['epoch'] for step in steps], 'value': [step['dev_loss'] for step in steps]}
    dev_accuracies = {'epoch': [step['epoch'] for step in steps], 'value': [step['dev_accuracy'] for step in steps]}
    
    save_plot(dev_losses, filename_prefix + '_dev_loss.png', 'Epoch vs Dev Loss', 'Epoch', 'Dev Loss')
    save_plot(dev_accuracies, filename_prefix + '_dev_accuracy.png', 'Epoch vs Dev Accuracy', 'Epoch', 'Dev Accuracy (%)')

log_file_path = "./data/iter/model_0/log"
steps = parse_log_file(log_file_path)
plot_data(steps, 'plots')
