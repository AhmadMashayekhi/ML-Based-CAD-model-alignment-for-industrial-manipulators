"""
=============================================================================
-------------------------------------INFO------------------------------------
=============================================================================



plotter (for info)

Contains different functions to plot results

"""

"""
=============================================================================
-----------------------------------IMPORTS-----------------------------------
=============================================================================
"""

#Import libraries
import matplotlib.pyplot as plt

"""----------------------------------------------------------------------------------"""
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PLOTTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
"""----------------------------------------------------------------------------------"""

def failure_division_plot(failure_div_list, nmb_scans, object_name):
    # Used to plot amount of failures for each scan
    scan_names = range(1,nmb_scans+1)
    
    fig1, ax1 = plt.subplots()
    ax1.bar(scan_names, failure_div_list)
    ax1.set_title("Failure division among scans for " + object_name)
    ax1.set_ylabel("Number of failures (%)")
    ax1.set_xlabel("Scan Number")
    ax1.legend()
    
def Acc_Loss_Plot(epoch_numb,train_losses,train_accuracies,test_losses,test_accuracies,exp_name,ths):
    fig1, ax1 = plt.subplots()
    ax1.plot(epoch_numb, train_losses,"orange",label="Training")
    ax1.plot(epoch_numb, test_losses,"blue",label ="Validation")
    ax1.set_title("Training and Validation Losses (Frobenius)")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Number of Epochs")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(epoch_numb, train_accuracies,"orange",label="Training")
    ax2.plot(epoch_numb, test_accuracies,"blue",label="Validation")
    ax2.set_title("Training and Validation Accuracies (loss < " + str(ths) + ")")
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_xlabel("Number of Epochs")
    ax2.legend()

    fig1.savefig('C:/Users/menth/Pictures/Thesis/Network Training/'+ exp_name +'_losses.eps', transparent=True)
    # fig2.savefig('C:/Users/menth/Pictures/Thesis/Network Training/'+ exp_name +'_accuracies.eps', transparent=True)
    
def read_run_file(DIR,exp_name):
    file = open(DIR, 'r')
    count = 0
    epoch_list = []
    train_losses = []
    train_acc = []
    test_losses = [] 
    test_acc = []
    
    lines = file.readlines()
    first_line = lines[0]
    lines = lines[1:]
    
    # exp_name = first_line.split('(')[1].split(',')[0].split('=')[1][1:-1]
    # print(exp_name)
    
    for line in lines:
        #print(line)
        count += 1
        L = line.split(',') #Split line at every ',' to get list with data
        
        epoch = int(L[0].split(': ')[-1])
        train_loss = float(L[1].split(': ')[-1])
        test_loss = float(L[2].split(': ')[-1])
        
        if(not((train_loss > 2 or test_loss > 2) and epoch > 0)):
            epoch_list.append(int(L[0].split(': ')[-1]))
            train_losses.append(float(L[1].split(': ')[-1]))
            test_losses.append(float(L[2].split(': ')[-1]))
            train_acc.append(float(L[4].split(': ')[-1])*100)
            test_acc.append(float(L[5].split(': ')[-1])*100)
  
    # Closing files
    file.close()
    #print(epoch_list)
    Acc_Loss_Plot(epoch_list, train_losses, train_acc, test_losses, test_acc, exp_name, 1)
    
    return 

if __name__ == '__main__':
    DIR = "C:/Users/menth/Documents/Python Scripts/Thesis/toolboxes/learning3d/checkpoints/run1.log"
    read_run_file(DIR,"overfitting")