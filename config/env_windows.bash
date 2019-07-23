# --- Configure environment

# Find top-level directory
TOP_DIR=`dirname "${BASH_SOURCE[0]}"`

# Data directory
export DATA_DIR=$TOP_DIR/data

# Set PATH
if [[ "$PATH" != *$TOP_DIR/bin* ]]; then
    export PATH=$TOP_DIR/bin:$PATH
fi

# Set PYTHONPATH
unset PYTHONPATH
if [[ "$PYTHONPATH" != *$TOP_DIR* ]]; then
    export PYTHONPATH=$TOP_DIR:$PYTHONPATH
fi
if [[ "$PYTHONPATH" != *$TOP_DIR/lib* ]]; then
    export PYTHONPATH=$TOP_DIR/lib:$PYTHONPATH
fi

# Jupyter configuration
export JUPYTER_CONFIG_DIR=$TOP_DIR/.jupyter

if [[ "$JUPYTER_PATH" != *$TOP_DIR/lib* ]]; then
    export JUPYTER_PATH=$TOP_DIR/lib:$JUPYTER_PATH
fi

# Jupyter aliases
alias jn='jupyter notebook'