#!/bin/bash



######################################################################
##
## GLOBALS
##
######################################################################
APP_NAME="LeNet"
CURRENT_DIR=.
REPORT_DIR=${CURRENT_DIR}/rept
LOG_DIR=${CURRENT_DIR}/logs
PROFILER_PATH=/usr/local/cuda/bin/nvprof
BIN=${CURRENT_DIR}/${APP_NAME}
REPORT_ID=$(date +%m-%d-%y_%H-%M-%S)
TMP_DIR=${CURRENT_DIR}/tmp
REPORT_FILE1=${REPORT_DIR}/${REPORT_ID}_light.rep
REPORT_FILE2=${REPORT_DIR}/${REPORT_ID}_medium.rep
REPORT_FILE3=${REPORT_DIR}/${REPORT_ID}_exhaustive.rep
STDERR_LOG=${LOG_DIR}/${REPORT_ID}_err.log

METRICS=""
METRICS+="gld_efficiency,gst_efficiency,gld_transactions,gst_transactions,"
METRICS+="shared_load_transactions_per_request,shared_store_transactions_per_request,shared_efficiency,"
METRICS+="achieved_occupancy,branch_efficiency"

EVENTS=""
EVENTS+="shared_ld_bank_conflict,shared_st_bank_conflict"


######################################################################
##
## SETUP
##
######################################################################
#check if BINARY exists
if [ ! -f $BIN ]; then
	echo "binary $BIN couldn't be found, compile your application with 'make'"
	exit 1
fi
#check if report dir exists
if [ ! -d $REPORT_DIR ]; then
	echo "report directory doesn't exist, i'm creating it"
	mkdir $REPORT_DIR
fi
#check if log dir exists
if [ ! -d $LOG_DIR ]; then
	echo "log directory doesn't exist, i'm creating it"
	mkdir $LOG_DIR
fi
#every new run requires to backup all logs and reports to tmp directory (this directory should not be versioned)
mkdir -p $TMP_DIR
mv ${REPORT_DIR}/* ${TMP_DIR} 2> /dev/null
mv -f ${LOG_DIR}/* ${TMP_DIR} 2> /dev/null


######################################################################
##
## 1: run profiler to generate lite report (no metrics, no events)
##
######################################################################
echo "Profiling [LITE] $APP_NAME.."
sudo $PROFILER_PATH $BIN  1>/dev/null 2>>$REPORT_FILE1
echo "Profile [LITE] saved in $REPORT_FILE1"

######################################################################
##
## 2: run profiler to generate medium detailed report
##
######################################################################
echo "Profiling [MEDIUM] $APP_NAME.."
sudo $PROFILER_PATH --metrics $METRICS --events $EVENTS $BIN  1>/dev/null 2>>$REPORT_FILE2
echo "Profile [MEDIUM] saved in $REPORT_FILE2"

######################################################################
##
## 3: run profiler to collect all metrics, all events
##
#####################################################################
echo "Profiling [EXHAUSTIVE] $APP_NAME.."
sudo $PROFILER_PATH --metrics all --events all $BIN  1>/dev/null 2>>$REPORT_FILE3
echo "Profile [EXHAUSTIVE] saved in $REPORT_FILE3"
