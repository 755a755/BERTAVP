import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from Bio import Seq, SeqIO


def read_fasta(fname):
    '''
    Read fasta file to dictionary
    Input: path name of fasta
    Output: dataframe of Peptide Seq {ID1: Seq1, ID2: Seq2,...}
    '''
    with open(fname, "rU") as f:
        seq_dict = [(record.id, record.seq._data) for record in SeqIO.parse(f, "fasta")]      #使用Bio.SeqIO.parse()函数读取fasta文件得到序列名字和内容，record会得到record.id和record.seq
    seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
    return seq_df


def insert_phycs(seq_df):  # 调用了modlamp包里面的PeptideDescriptor去计算hydrophobic moment，Hydrophobicity，Transmembrane Propensity，Alpha Helical Propensity
    seq_df = seq_df.copy()  # 调用了GlobalDescriptor去计算Aromacity，Aliphatic Index，Boman Index

    #  Function for compute Isoelectric Point or net_charge of peptide
    def get_ieq_nc(seq, is_iep=True):
        protparam = PA(seq)
        return protparam.isoelectric_point() if is_iep else protparam.charge_at_pH(7.0)

    # Calculating IsoElectricPoints and NeutralCharge
    data_size = seq_df.size
    seq_df['PHYC|IEP'] = list(map(get_ieq_nc, seq_df['Sequence'], [True] * data_size))  # IsoElectricPoints
    seq_df['PHYC|Net Charge'] = list(map(get_ieq_nc, seq_df['Sequence'], [False] * data_size))  # Charge(Neutral)

    # Calculating hydrophobic moment (My assume all peptides are alpha-helix)
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'eisenberg')
    descrpt.calculate_moment(window=1000, angle=100, modality='max')
    seq_df['PHYC|Hydrophobic Moment'] = descrpt.descriptor.reshape(-1)

    # Calculating "Hopp-Woods" hydrophobicity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'hopp-woods')
    descrpt.calculate_global()
    seq_df['PHYC|Hydrophobicity'] = descrpt.descriptor.reshape(-1)

    # Calculating Energy of Transmembrane Propensity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'tm_tend')
    descrpt.calculate_global()
    seq_df['PHYC|Transmembrane Propensity'] = descrpt.descriptor.reshape(-1)

    # Calculating Aromaticity
    descrpt = GlobalDescriptor(seq_df['Sequence'].values)
    descrpt.aromaticity()
    seq_df['PHYC|Aromacity'] = descrpt.descriptor.reshape(-1)

    # Calculating Levitt_alpha_helical Propensity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'levitt_alpha')
    descrpt.calculate_global()
    seq_df['PHYC|Alpha Helical Propensity'] = descrpt.descriptor.reshape(-1)

    # Calculating Aliphatic Index
    descrpt = GlobalDescriptor(seq_df['Sequence'].values)
    descrpt.aliphatic_index()
    seq_df['PHYC|Aliphatic Index'] = descrpt.descriptor.reshape(-1)

    # Calculating Boman Index
    descrpt = GlobalDescriptor(seq_df['Sequence'].values)
    descrpt.boman_index()
    seq_df['PHYC|Boman Index'] = descrpt.descriptor.reshape(-1)

    return seq_df

import re, sys, os
from collections import Counter
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import re, sys, os, platform
import math
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import numpy as np
import pandas as pd


def readFasta(file):
	if os.path.exists(file) == False:
		print('Error: "' + file + '" does not exist.')
		sys.exit(1)

	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input file seems not in fasta format.')
		sys.exit(1)

	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta

def checkFasta(fastas):
	status = True
	lenList = set()
	for i in fastas:
		lenList.add(len(i[1]))
	if len(lenList) == 1:
		return True
	else:
		return False

def minSequenceLength(fastas):
	minLen = 10000
	for i in fastas:
		if minLen > len(i[1]):
			minLen = len(i[1])
	return minLen

def minSequenceLengthWithNormalAA(fastas):
	minLen = 10000
	for i in fastas:
		if minLen > len(re.sub('-', '', i[1])):
			minLen = len(re.sub('-', '', i[1]))
	return minLen

def AAC(fastas, **kw):
	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])	#将sequence里的‘-’换成了‘’。
		count = Counter(sequence)								#算序列字母的个数，比如L有2个，K有2个
		for key in count:
			count[key] = count[key]/len(sequence)					#conunt[key]原本是存着序列字母的个数，但是经过这一句之后就变成了count[key]上的key氨基酸所占的比例（概率）
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)
	return encodings

def CKSAAP(fastas, gap=5, **kw):
	if gap < 0:
		print('Error: the gap should be equal or greater than zero' + '\n\n')
		return 0

	if minSequenceLength(fastas) < gap+2:
		print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap+2) + '\n\n')
		return 0

	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	aaPairs = []
	for aa1 in AA:
		for aa2 in AA:
			aaPairs.append(aa1 + aa2)
	header = ['#']                              #aaPairs存着A氨基酸从A氨基酸一直配对到Y氨基酸，然后开始C氨基酸的配对一直配到Y氨基酸，即所有可能性的残基对，400个
	for g in range(gap+1):						#g是0到5，这两个for嵌套只是将aaPairs里残基对加上.gap+str（g）写到header里，最后写到encodings
		for aa in aaPairs:
			header.append(aa + '.gap' + str(g))
	encodings.append(header)                   #写了6组所有可能性的残基对进去。
	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for g in range(gap+1):
			myDict = {}
			for pair in aaPairs:
				myDict[pair] = 0                     #将myDict字典里的pair初始化，就是将400个残基对都初始化为0
			sum = 0
			for index1 in range(len(sequence)):
				index2 = index1 + g + 1
				if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA: #应该是只有index2有机会不符合这个if条件
					myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
					sum = sum + 1																							#当g=0时（间隔为0），找序列里间隔为0的残基对，找到一个就在myDict里+1，
			for pair in aaPairs:																							#算完g=0时（算完间隔为0的残基对数量时），通过遍历全部残基对并除总数放到code里
				code.append(myDict[pair] / sum)																				#算完g=0时，就算g=1（间隔为1的残基对数量）
		encodings.append(code)																								#一直算到g=5时，
	return encodings																										#g=0，myDict里有400个总的残基对，g=1时，也有400个残基对，但是第一个会有name写着


def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=6, w=0.5, **kw):
	if minSequenceLengthWithNormalAA(fastas) < lambdaValue + 1:
		print('Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
		return 0

	dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data/PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}                                    #从文件PAAC.txt里读取数据，将“ARNDCQEGHILKMFPSTWYV”读出来并赋值给AA
	for i in range(len(AA)):
		AADict[AA[i]] = i							#应该是创建以AA氨基酸名字的字典并赋上0~19,'ARNDCQEGHILKMFPSTWYV'{’A‘:0，'R':1,'N':2....}
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):													#这个for只是将PAAC文件里的疏水性和亲水性特性督导AAProperty和AAPropertyNames
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])
		#AAProperty里面装着都是PAAC文件序列特征后面的数据，AAPropertyNames里装的都是三个特征名字
	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])
	#AAProperty1里存了三个特征经过公式算出来以后的值，每一个Xj-meanI/fenmu的值（Xj是AAProperty里的值）
	encodings = []
	header = ['#']
	for aa in AA:
		header.append('Xc1.' + aa)
	for n in range(1, lambdaValue + 1):      #n是从1到6
		header.append('Xc2.lambda' + str(n))
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		theta = []
		for n in range(1, lambdaValue + 1): #lambdaValue=6，n是1到6  #seq:'LKAKTNISIREGPTLGNWAR'
			theta.append(
				sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
				len(sequence) - n))
		myDict = {}
		for aa in AA:
			myDict[aa] = sequence.count(aa)
		code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
		code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
		encodings.append(code)
	return encodings

'''def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)'''

kw = {'path': r"C:",'order': 'ACDEFGHIKLMNPQRSTVWY'}
# fastas = readFasta(r"./dataset/main dataset/first/firstStage.faa")
fastas = readFasta(r"./dataset/main dataset/second/secondStage.faa")
aac = AAC(fastas, **kw)
cksaap = CKSAAP(fastas, 5, **kw)  #FFMAVP数据集都是gap=5，但540因为序列长度短所以gap改成4
paac = PAAC(fastas, 4, 0.05,  **kw)

data_AAC = np.matrix(aac[1:])[:, 1:]
data_AAC = pd.DataFrame(data=data_AAC)

data_CKSAAP = np.matrix(cksaap[1:])[:, 1:]
data_CKSAAP = pd.DataFrame(data=data_CKSAAP)

data_PAAC = np.matrix(paac[1:])[:, 1:]
data_PAAC = pd.DataFrame(data=data_PAAC)

seq_df = read_fasta('./dataset/main dataset/second/secondStage.faa')
df = insert_phycs(seq_df)
df = pd.DataFrame(df)
df = np.array(df.values)
df = df[:, 2:]
feature = np.column_stack((data_AAC,data_CKSAAP,data_PAAC,df))#AAC是20维度，CSKAAP是2000维度(gap=4)，gap=5为2400，PAAC是24维度，phy9是9维度。#(data_AAC,data_CKSAAP,data_PAAC,df)
feature = pd.DataFrame(feature)
print(feature)
feature.to_csv("./test_second.csv", header=True, index=False)

#------------------要用AVPIden那个环境才能跑成功，不然会报错!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------

