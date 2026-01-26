from api.db.crud.predictions import create_new_prediction


def test_create_new_prediction(db):
    user_id = '1234'
    task_type = 'classification'
    best_model = 'LinReg'
    target = 'Survived'
    metric = '1.000'
    dataset_hash = 'data_hash'

    prediction = create_new_prediction(db, user_id, task_type, best_model,
                                        target, metric, dataset_hash)

    assert prediction is not None
    assert prediction.best_model == 'LinReg'
