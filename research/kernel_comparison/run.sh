# usage: run.sh <kernel> <method> <parallel>
K=/tmp/claude-0/-home-user/4f06e2ce-f8b8-5274-beb2-32b879306b77/scratchpad/kern
A=/tmp/claude-0/-home-user-SberAuto/4f06e2ce-f8b8-5274-beb2-32b879306b77/scratchpad/src/venv/bin/ai4epi
cd /home/user/ai_epidemiologist
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KERNEL=$1 METHOD=$2
cat $K/seasons.txt | xargs -P $3 -L 1 sh -c 'y=$1; s=$(python3 -c "import datetime as d;t=d.date($y,10,1);print(t+d.timedelta(days=(7-t.weekday())%7))"); e=$(python3 -c "import datetime as d;t=d.date($y+1,4,30);print(t-d.timedelta(days=t.weekday()))"); t0=$(date +%s); timeout 7000 '$A' backtest-local --city $0 --start $s --end $e --br-method $METHOD --br-kernel $KERNEL --warmup-weeks 0 --n-jobs 1 --output-dir '$K'/out_${KERNEL}_${METHOD}/$0_$y > '$K'/log_${KERNEL}_${METHOD}_$0_$y.txt 2>&1; echo "$0 $y exit $? $(( $(date +%s)-t0 ))s" >> '$K'/status_${KERNEL}_${METHOD}.txt'
