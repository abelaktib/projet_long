### Save only best model


mcp_save = ModelCheckpoint(file_out , save_best_only=True, monitor='val_loss', mode='min')

    # Fitting the model
history = model.fit([data_input, mask], data_output, validation_split=0.2,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks= [mcp_save])