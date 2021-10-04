rm -f out_asdf* 
for i in {1..1}
do
	python test_multiprocessing.py > out_asdf$i.txt &
done
