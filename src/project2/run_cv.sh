OLD_VALUE=1
for NEW_VALUE in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1
do
    # Shuffle and produce training and test sets
    shuf /home/hduser/public/training > /home/hduser/training-shuffled
    head -n 10000 /home/hduser/training-shuffled > /home/hduser/training-test
    tail -n 90000 /home/hduser/training-shuffled > /home/hduser/training-train
    # Modify the mapper with the parameters
    sed -i "s/_REGULARIZATION=${OLD_VALUE}/_REGULARIZATION=${NEW_VALUE}/g" /home/hduser/public/mapper.py
    OLD_VALUE=$NEW_VALUE 
    grep _REGULARIZATION /home/hduser/public/mapper.py
done
