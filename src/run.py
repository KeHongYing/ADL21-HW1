import subprocess

hidden_layer = [512]
num_layer = [4, 5]
init_parm = [True, False]

for h in hidden_layer:
    for n in num_layer:
        for init in init_parm:
            status, output = subprocess.getstatusoutput(f"python src/train_intent.py --device cuda:4 --hidden_size {h} --num_layer {n} --init_parm {init} --model report_hidden_{h}_num_layer_{n}{'_init' if init else ''}")
            if status != 0:
                print(output)
                print("intetn error")
                exit(0)

            status, output = subprocess.getstatusoutput(f"python src/train_slot.py --device cuda:4 --hidden_size {h} --num_layer {n} --init_parm {init} --model report_hidden_{h}_num_layer_{n}{'_init' if init else ''}")
            if status != 0:
                print(output)
                print("slot error")
                exit(0)