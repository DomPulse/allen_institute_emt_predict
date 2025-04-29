import bamnostic as bs
test_file_path = r'F:\Big_MET_data\morpho_trans_extracted\ar_657938511_STAR_Aligned.sortedByCoord.out.bam'
bam = bs.AlignmentFile(test_file_path, 'rb')
print(bam.header)