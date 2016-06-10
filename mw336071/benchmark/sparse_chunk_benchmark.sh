
for m in 0 2
do
	iter=1
	#for iter in `seq 1 2`
	#do
		for f in `ls benchmark_sparse/* | sort -r`
		do
			for c in 1 2 4
			do
				echo m: $m iter: $iter f: $f c: $c
				time mpirun -np 8 ./matrixmul -f $f -s 456 -c $c -e 2 -m $m -x >> result.csv
			done
		done
	#done
done
