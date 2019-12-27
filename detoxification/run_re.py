import os

if __name__ == '__main__':
    tasks = {}
    for line in open('./compromised.txt'):
        if line.startswith('weights name'):
            key = line.split()[3]
            print('model_weights', key)
            tasks[key] = []
        elif line.startswith('Layer:'):
            layer = line.split()[1]
            neuron = line.split()[3]
            label = '0'
            tasks[key].append([layer, neuron, label])

    if not os.path.exists('./trigger_imgs'):
        os.system('mkdir ./trigger_imgs')
    if not os.path.exists('./trigger_pkls'):
        os.system('mkdir ./trigger_pkls')
    if os.path.exists('result.txt'):
        os.remove('result.txt')
    os.system('rm -r ./trigger_imgs/*')
    os.system('rm -r ./trigger_pkls/*')

    # *** Several Reverse Engineering Functions ***
    trigger_option = 1
    for key in sorted(tasks.keys()):
        weights_file = key
        for task in tasks[key]:
            Troj_Layer, Troj_Neuron, Troj_Label = task

            flog = open('result.txt', 'a')
            flog.write('\n\n{0} {1} {2} {3}\n\n'.format(weights_file, Troj_Layer, Troj_Neuron, Troj_Label))
            flog.close()

            os.system('python detoxification.py {0} {1} {2} {3} {4} {5}'.format('./dataset/seed_test', weights_file,
                                                                                  Troj_Layer, Troj_Neuron, Troj_Label,
                                                                                  trigger_option))
