import bamnostic as bs

test_file_path = r'F:\Big_MET_data\morpho_trans_extracted\ar_657938511_STAR_Aligned.sortedByCoord.out.bam'

bam = bs.AlignmentFile(test_file_path, 'rb')

count_dracula = {}

read_count = 0
for read in bam:
	the_whole_tomale = f"{read.reference_name}-{read.reference_start}"
	print(the_whole_tomale)
	if (the_whole_tomale not in count_dracula):
		count_dracula[the_whole_tomale] = 1
	else:
		count_dracula[the_whole_tomale] += 1

print(count_dracula)

