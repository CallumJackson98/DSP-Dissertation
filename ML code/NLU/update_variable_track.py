import csv


def update(run_info):
    allData = []
    
    headers = ['final loss', 'learning rate', 'num_epochs', 'batch_size', 'loss_track']
    with open('E:\\Uni Work\\Uni\\Year 3\\Digital Systems Project\\tests\\pytorch_tests\\NLU_test_with_dataset\\track_variables.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        next(reader)
        for r in reader:
            allData.append([float(r[0]), float(r[1]), float(r[2]), float(r[3]), r[4]])
        
    allData.append(run_info)
    
    allData.sort()
    
    with open('E:\\Uni Work\\Uni\\Year 3\\Digital Systems Project\\tests\\pytorch_tests\\NLU_test_with_dataset\\track_variables.csv', 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(headers)
        for a in allData:
            writer.writerow(a)
    