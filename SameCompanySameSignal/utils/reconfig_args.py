def reconfig_args(args):
    args.output_dir = ''
    args.y_column = f'lv{args.prediction_window}_future_{args.prediction_window}'
    if args.data_path in ['EC', 'MAEC15', 'MAEC16']:
        args.y_column = f"future_{args.prediction_window}"

    args.model_id +=  f"_win{args.prediction_window}"

    args.rolling_test = f"{args.test_year}_{args.test_quarter}"
    args.model_id +=  f"_rt({args.test_year}|{args.test_quarter})"

    if args.task_name == 'text_earnings':
        if args.emb_vendor != 'openai':
            args.output_dir += args.emb_vendor + '/' 
        
        args.output_dir += args.emb_file + '/'  

    if args.task_name == 'ts_earnings':
        pass
        
    return args
            