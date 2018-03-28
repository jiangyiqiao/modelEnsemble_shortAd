#encoding:UTF-8

filenames=["ad.txt","not_ad.txt"]
output=open("ensemble.txt","w")

for filename in filenames:
    with open(filename, 'r') as f:
        for line in f:   
            if filename=="ad.txt":  
                output.write("ad\t"+line)
            else:
                output.write("not_ad\t"+line)
    f.close()

output.flush()
output.close()








