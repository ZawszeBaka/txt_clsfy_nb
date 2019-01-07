
import numpy as np #su dung thu vien numpy
#cac du lieu co san
Z=np.array([("stt","mau sac","dong xe","xuat xu","nguoi mua"),
("1","do","the thao","trong nuoc","nam"),
("2","do","the thao","trong nuoc","nu"),
("3","do","the thao","trong nuoc","nam"),
("4","vang","the thao","trong nuoc","nu"),
("5","vang","the thao","nhap","nam"),
("6","vang","du lich","nhap","nu"),
("7","vang","du lich","nhap","nam"),
("8","vang","du lich","trong nuoc","nu"),
("9","do","du lich","nhap","nu"),
("10","do","the thao","nhap","nam")])
#tao ma tran a
a=np.array(["0","0000","00000000","0000000000"])
#nhap cac dac diem nguoi mua
print("nhap mau sac")
a[1]=input()
print("nhap dong xe")
a[2]=input()
print("nhap xuat xu")
a[3]=input()
print(a)
#tinh so diem du lieu cho truoc co nguoi mua la nam
nc=np.array([0,0,0,0])
for j in range(1,4):
    for i in range(1,11):
        if Z[i,4]=="nam" and Z[i,j]==a[j]:
            nc[j]=nc[j]+1.0
#tinh xac xuat uoc luong tien dinh
n=0
for i in range(1,11):
    if Z[i,4]=="nam":
        n=n+1.0
p1=pnam=n/(Z.shape[0]-1.0)#tong so du lieu bang so hang -1
p2=pnu=1-pnam
m=Z.shape[1]-2#so hang doc -2
#xac suat nguoi mua la nam
for i in range(1,4):
    pnam=pnam*(nc[i]+m*p1)/(n+m)
print("pnam la");print(pnam)
#so diem du lieu cho truoc co nguoi mua la nu
nc=np.array([0,0,0,0])
for j in range(1,4):
    for i in range(1,11):
        if Z[i,4]=="nu" and Z[i,j]==a[j]:
            nc[j]=nc[j]+1.0
#xac suat nguoi mua la nu
for i in range(1,4):
    pnu=pnu*(nc[i]+m*p2)/(n+m)
print("pnu la");print(pnu)
#so sanh xac suat nguoi mua la nam va nguoi mua la nu
if pnam>pnu:
    print("do la nam")
elif pnu>pnam:
    print("do la nu")
else:
    print("khong xac dinh")
