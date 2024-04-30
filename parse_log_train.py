import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    steps = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if line.startswith('Step'):
                parts = line.split(',')
                if parts[0][-3:] == '000':
                    step_info = {}
                    step_info['step'] = int(parts[0].split()[1])
                    step_info['train_loss'] = float(parts[1].split(':')[1])
                    step_info['accuracy'] = float(parts[2].split(':')[1].strip().split()[0])
                    steps.append(step_info)
    return steps

def save_plot(data, filename, title, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(data['step'], data['value'], marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_data(steps, filename_prefix):
    train_losses = {'step': [step['step'] for step in steps], 'value': [step['train_loss'] for step in steps]}
    accuracies = {'step': [step['step'] for step in steps], 'value': [step['accuracy'] for step in steps]}
    
    save_plot(train_losses, filename_prefix + '_train_loss.png', 'Step vs Train Loss', 'Step', 'Train Loss')
    save_plot(accuracies, filename_prefix + '_accuracy.png', 'Step vs Accuracy', 'Step', 'Accuracy (%)')

log_file_path = "./data/iter/model_0/log"
steps = parse_log_file(log_file_path)
plot_data(steps, 'plots')
