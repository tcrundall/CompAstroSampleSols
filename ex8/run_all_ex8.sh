for NPARTS in 100 500 1000 
do
  for ALPHA in 1 0.1 0.01 
  do
  OUTFILE="${NPARTS}_${ALPHA}.out"
  echo $OUTFILE
  # echo 'sup' >> $OUTFILE
  python ex8.py $NPARTS $ALPHA > $OUTFILE
  done
done

