echo "
###########################################################################################################
#                                           CNN workstation                                               #
###########################################################################################################
"

echo "Starting CNN workstation..."

while true; do
    echo "
    Enter your mode: 
    1. Start a new training session (train)
    2. Get prediction result based on the best model (test)
    3. Straight to portfolio forming (pass)
    "
    read -p "Enter your mode (train / test / pass): " mode

    echo "
    Enter the type of label:
    1. bin (binary)
    2. multi (multi-class)
    "
    read -p "Enter the type of model (bin / multi): " mdl
    
    if [ "$mode" = "train" ]; then
        python -m src.cnn_workstation_v2 --mode $mode --mdl $mdl
        break

    elif [ "$mode" = "test" ]; then
        echo "
        Enter the labeler:
        1. bin: binary classification
        2. multi-h: decile classification with 10 labels (highest label prob)
        3. multi-l: decile classification with 10 labels (lowest label prob)
        4. ova-h: one-vs-all classification with highest label as 1
        5. ova-l: one-vs-all classification with lowest label as 1
        "
        read -p "Enter the labeler: " labeler

        python -m src.cnn_workstation_v2 \
            --mode $mode \
            --labeler $labeler \
            --mdl $mdl
        break

    elif [ "$mode" = "pass" ]; then
        echo "Pass to portfolio forming."
        break

    else
        echo "Invalid mode, please try again."
    fi
done

echo "
##########################################################################################################
#                                             Portfolio Former                                           #
##########################################################################################################
"
read -p "Form a portfolio? (y/n) " to_form_portfolio

if [ $to_form_portfolio = "y" ]; then
    if [ $mode = "pass" ]; then
        echo "
        Enter the labeler:
        1. bin: binary classification
        2. decile: decile classification with 10 labels
        3. ova-h: one-vs-all classification with highest label as 1
        4. ova-l: one-vs-all classification with lowest label as 1
        "
        read -p "Enter the labeler: " labeler
        echo "
        Enter the portfolio forming method: 
        1. equal: equal weight for each stock
        2. value: weight based on the value of each stock
        "
        read -p "Enter the portfolio forming method: (type the name) " portfolio_forming_method

        
        python -m src.portfolio \
            --model_opt_path dat/pred_data/predict_label_$labeler.csv \
            --SHOW True \
            --type $portfolio_forming_method
    else
        echo "
        Enter the portfolio forming method: 
        1. equal: equal weight for each stock
        2. value: weight based on the value of each stock
        "
        read -p "Enter the portfolio forming method: (type the name) " portfolio_forming_method

        
        python -m src.portfolio \
            --model_opt_path dat/pred_data/predict_label_$labeler.csv \
            --SHOW True \
            --type $portfolio_forming_method
    fi 
elif [ $to_form_portfolio = "n" ]; then
    echo "No portfolio is formed."
fi 