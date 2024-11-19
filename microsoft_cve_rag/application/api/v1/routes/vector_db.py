from fastapi import APIRouter, HTTPException, Depends, Query
from application.core.schemas.vector_schemas import (
    VectorRecordCreate,
    VectorRecordUpdate,
    VectorRecordResponse,
    VectorRecordQuery,
    VectorRecordQueryResponse,
    BulkVectorRecordCreate,
    BulkVectorRecordDelete,
)
from application.services.vector_db_service import VectorDBService
from application.services.llama_index_service import LlamaIndexVectorService
from application.app_utils import get_app_config
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
import os
import asyncio
from typing import Optional, List, Dict, Union
import logging

logging.getLogger(__name__)

router = APIRouter()


async def get_vector_db_service():
    settings = get_app_config()
    vector_db_settings = settings["VECTORDB_CONFIG"]
    embedding_settings = settings["EMBEDDING_CONFIG"]

    service = VectorDBService(
        collection=vector_db_settings["tier1_collection"],
        distance_metric=vector_db_settings["distance_metric"],
        embedding_config=embedding_settings,
        vectordb_config=vector_db_settings,
    )

    try:
        yield service
    finally:
        await service.aclose()

# Add dependency function at the top with the other dependencies
async def get_llama_index_service():
    settings = get_app_config()
    vector_db_settings = settings["VECTORDB_CONFIG"]
    embedding_settings = settings["EMBEDDING_CONFIG"]
    persist_dir = vector_db_settings["persist_dir"]

    vector_service = VectorDBService(
        collection=vector_db_settings["tier1_collection"],
        distance_metric=vector_db_settings["distance_metric"],
        embedding_config=embedding_settings,
        vectordb_config=vector_db_settings,
    )

    try:
        llama_service = await LlamaIndexVectorService.initialize(
            vector_service,
            persist_dir
        )
        yield llama_service
    finally:
        await llama_service.aclose()

@router.post("/vectors/", response_model=VectorRecordResponse)
async def create_vector(
    vector: VectorRecordCreate,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
) -> VectorRecordResponse:
    """
    Create a new vector.

    Args:
        vector (VectorRecordCreate): The vector data to create.

    Returns:
        VectorRecordResponse: The response containing the vector ID and status message.
    """
    try:

        result = await vector_db_service.create_vector(vector)
        return VectorRecordResponse(
            id=result["point_id"],
            message="Vector created successfully",
            status=result["status"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectors/{vector_id}", response_model=VectorRecordResponse)
async def get_vector(
    vector_id: str,
    with_payload: Union[bool, List[str]] = Query(True),
    with_vectors: Union[bool, List[str]] = Query(True),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
) -> VectorRecordResponse:
    """
    Retrieve a vector by its ID.

    Args:
        vector_id (str): The ID of the vector to retrieve.
        with_payload (Union[bool, List[str]]): Payload keys to retrieve. Defaults to True.
        with_vectors (bool): Whether to retrieve the vector. Defaults to True.

    Returns:
        VectorRecordResponse: The response containing the vector data and status message.
    """
    try:
        # Handle both boolean and list inputs
        with_payload_actual = with_payload if isinstance(with_payload, bool) else with_payload[0]
        with_vectors_actual = with_vectors if isinstance(with_vectors, bool) else with_vectors[0]

        if isinstance(with_payload_actual, str):
            with_payload_actual = with_payload_actual.lower() == "true"
        if isinstance(with_vectors_actual, str):
            with_vectors_actual = with_vectors_actual.lower() == "true"

        result = await vector_db_service.get_vector(
            vector_id,
            with_payload=with_payload_actual,
            with_vectors=with_vectors_actual,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Vector not found")
        return VectorRecordResponse(
            id=result.id,
            message="Vector retrieved",
            vector=result.vector,
            payload=result.payload,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/vectors/{vector_id}", response_model=VectorRecordResponse)
async def update_vector(
    vector_id: str,
    vector: VectorRecordUpdate,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Update an existing vector by its ID. If text is passed, the service will automatically recompute the vector embeddings and overwrite the existing text associated with the vector. VectorRecordUpdate defines the available metadata keys that can be included in the request.

    Args:
        vector_id (str): The ID of the vector to update.
        vector (VectorRecordUpdate): The updated vector data.

    Returns:
        VectorRecordResponse: The response containing the updated vector_id and message.
    """
    try:
        response = await vector_db_service.update_vector(vector_id, vector)
        print(f"{type(response)}: {response}")
        return VectorRecordResponse(
            id=vector_id,
            message=f"Vector updated successfully. {response['status']}.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectors/{vector_id}", response_model=VectorRecordResponse)
async def delete_vector(
    vector_id: str, vector_db_service: VectorDBService = Depends(get_vector_db_service)
):
    """
    Delete a vector by its ID.

    Args:
        vector_id (str): The ID of the vector to delete.

    Returns:
        VectorRecordResponse: The response containing the deleted vector data and status message.
    """
    try:
        response = await vector_db_service.delete_vector(vector_id)

        return VectorRecordResponse(
            id=str(response["operation_id"]),
            message=f"Vector deleted successfully. {response['status']}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/points/{vector_id}", response_model=VectorRecordResponse)
async def delete_point(
    vector_id: Optional[str] = None,
    metadata_id: Optional[str] = None,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Delete a point by its vector ID or metadata ID. One is required.

    Args:
        vector_id (Optional[str]): The ID of the vector to delete.
        metadata_id (Optional[str]): The metadata ID to delete.

    Returns:
        VectorRecordResponse: The response containing the deleted point data and status message.
    """
    print(f"vector_id: {vector_id}")
    if not vector_id and not metadata_id:
        raise ValueError("Must pass either vector_id or metadata.id")
    try:
        response = await vector_db_service.delete_point(
            vector_id=vector_id, metadata_id=metadata_id
        )

        return VectorRecordResponse(
            id=str(response["operation_id"]),
            message=f"Point deleted successfully. {response['status']}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vectors/search", response_model=List[VectorRecordQueryResponse])
async def search_vectors(
    query: VectorRecordQuery,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Search for vectors based on a query.

    Args:
        query (VectorRecordQuery): The search query parameters.

    Returns:
        List[VectorRecordQueryResponse]: A list of vector records matching the query.
    """
    try:
        results = await vector_db_service.search_vectors(query.text, query.limit)
        return [
            VectorRecordQueryResponse(id=r.id, score=r.score, payload=r.payload)
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/bulk", response_model=List[VectorRecordResponse])
async def bulk_create_vectors(
    vectors: BulkVectorRecordCreate,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Create multiple vectors in bulk.

    Args:
        vectors (BulkVectorRecordCreate): The list of vectors to create.

    Returns:
        List[VectorRecordResponse]: A list of responses for each created vector.
    """
    try:
        results = await vector_db_service.bulk_create_vectors(vectors.vectors)
        return [
            VectorRecordResponse(id=r, message="Vector created successfully")
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/bulk-delete", response_model=Dict[str, int])
async def bulk_delete_vectors(
    vector_ids: BulkVectorRecordDelete,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Delete multiple vectors in bulk.

    Args:
        vector_ids (BulkVectorRecordDelete): The list of vector IDs to delete.

    Returns:
        Dict[str, int]: A dictionary containing the number of successfully deleted vectors.
    """
    try:
        deleted_count = await vector_db_service.bulk_delete_vectors(vector_ids.ids)
        return {"deleted_count": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/synced/points-delete", response_model=VectorRecordResponse)
async def delete_all_points(
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Delete all points from the collection.
    
    This operation:
    1. Gets all points from the collection
    2. Deletes them
    """
    try:
        result = await vector_db_service.delete_all_points()
        
        return VectorRecordResponse(
            id=str(result["operation_id"]),
            message=result["message"],
            status=result["status"]
        )
            
    except Exception as e:
        logging.error(f"Error deleting all points: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete all points: {str(e)}"
        )

@router.get("/analysis/uniqueness", response_model=Dict[str, Union[int, Dict[str, int]]])
async def get_uniqueness_analysis(
    llama_service: LlamaIndexVectorService = Depends(get_llama_index_service),
):
    """
    Analyze the uniqueness of points in the collection.
    
    Returns:
        Dict[str, int]: Counts of total and unique points based on different criteria
    """
    try:
        unique_counts = await llama_service._count_unique_points()
        return unique_counts
    except Exception as e:
        logging.error(f"Error analyzing point uniqueness: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze point uniqueness: {str(e)}"
        )
        
@router.post("/analysis/get-points-with-vector", response_model=List[Dict])
async def get_points_with_vector(vector_db_service: VectorDBService = Depends(get_vector_db_service),
 ) -> List[Dict]:
     """
     Get all points that match a specific vector and write to a text file.
 
     Args:
         vector (List[float]): The vector to search for.
 
     Returns:
         List[Dict]: The points matching the vector.
     """
     vector=[0.024737226, -0.038967624, -0.0077721816, -0.0444003, -0.020414725, -0.015111201, 0.07127667, 0.0044120844, 0.008693449, -0.0327935, 0.017670894, 0.02304502, -0.0077759614, 0.0014443737, -0.028781112, 0.04018888, -0.014383082, -0.026809944, -0.021084918, 0.030295625, 0.018017653, -0.032952666, 0.017635003, -0.016955992, 0.0035155877, 0.046408053, 0.014292216, -0.043443915, 0.041471615, -0.048319627, -0.03396268, 0.0014712048, 0.018659096, 0.0028838497, -0.015194841, -0.04428967, 0.011157752, -0.055224966, 0.028372217, -0.0070327846, 0.0075477855, 0.02784495, -0.029211298, 0.05742272, 0.0072350195, -0.011365984, -0.028028725, 0.028999198, 0.033590984, 0.0040586987, 0.03174295, -0.029470675, 0.016333811, -0.064327955, -0.00860768, 0.024802394, 0.019931326, 0.02367617, 0.020834697, 0.018293876, -0.042418, -0.0042141853, -0.047460057, -0.0047097174, -0.01116907, 0.053616565, 0.03295674, -0.006201015, 0.03896403, 0.0422722, 0.0006126188, -0.005848114, -0.014393951, 0.0028757001, 0.056391113, -0.041407574, -0.010770881, -0.010318529, 0.0039853416, 0.02440956, -0.029686516, 0.006710689, 0.023826407, 0.018449735, 0.05222285, -0.013897447, 0.084201254, -0.0060159992, -0.013464154, -0.013914123, -0.026521562, -0.029412344, 0.037932873, -0.009503667, -0.0503143, 0.0416563, -0.024854932, -0.035485, 0.0020944593, 0.013597293, 0.01327117, 0.010967454, 0.017435502, 0.05903854, 0.021353623, 0.022182122, -0.0152993575, 0.016360713, -0.025319662, -0.012214139, -0.026584454, 0.01973074, -0.070568286, -0.024536217, 0.024452677, -0.050270908, -0.035146363, -0.02437251, -0.039985623, -0.03079474, -0.0038134567, 0.05378216, 0.0065066186, -0.08123026, 0.02778807, 0.026660534, 0.070555985, -0.069477536, 0.006617516, 0.031126391, 0.009594648, -0.03246045, 0.03057748, -0.02133814, -0.019061202, 0.03884077, -0.08521711, -0.048034716, 0.044890784, 0.020967806, 0.012467962, -0.016076392, 0.028492894, 0.035631295, 0.021013366, 0.031918686, 0.019748738, 0.00015257226, -0.0441479, -0.017005656, -0.01592072, 0.01506934, -0.060038757, -0.030910535, -0.04389981, -0.029143672, 0.015568532, 0.012419166, -0.016912598, -0.042034443, -0.021719286, 0.006691748, 0.035128556, -0.02332295, -0.01942432, -0.029490035, 0.007825756, -0.02356941, 0.009030581, 0.04332847, -0.05417797, -0.027972452, -0.006761502, -0.00077550183, -0.006043284, 0.034051947, -0.016170567, -0.042366978, 0.006710241, -0.0646914, -0.02400324, -0.01856291, -0.07843839, 0.04694124, 0.05287219, 0.007981851, -0.013109836, -0.04507825, 0.016416527, 0.014729555, 0.020412501, -0.04260131, 0.008619881, -0.00657692, -0.031287476, 0.013413375, -0.04796753, 0.027973888, 0.012793293, 0.0044115684, 0.0558454, -0.03730692, 0.03964149, -0.047431137, -0.04092501, -0.013278395, -0.007462649, -0.024720648, -0.007424022, 0.005554867, -0.019400993, 0.02989894, 0.0092411665, 0.02591115, -0.025120119, 0.024422383, -0.01764599, 0.06329991, 0.016115464, -0.03530584, -0.061132744, -0.00839622, 0.03636742, 0.071149595, -0.019239092, -0.029086167, 0.03031692, -0.05747192, 0.022569865, 0.004057972, 0.021298647, -0.028181206, 0.025728468, -0.02218587, -0.0066594845, 0.052771896, 0.045115035, -0.019599857, -0.0118632065, -0.01082515, 0.030676173, 0.024660597, 0.061336398, 0.017791472, 0.04309144, 0.0052638375, -0.017335521, 0.0027067026, -0.019293353, 0.04464114, 0.008091253, 0.0067846472, 0.043315038, -0.06670528, -0.026357275, 0.0093194675, -0.009709456, 0.012684364, -0.007060572, 0.025778702, -0.04919522, -0.00025203198, 0.02562233, 0.00929155, 0.039632246, 0.046165615, 0.025710313, -0.018321158, 0.06049422, -0.05344228, 0.015668593, -0.015509482, -0.051440697, 0.001863335, -0.0015033309, 0.048446044, -0.068151236, 0.0166086, -0.02128558, -0.023863576, -0.0050104843, -0.011574095, -0.0051096585, 0.00432837, -0.00877896, 0.009828591, 0.01579215, 0.01456913, 0.034937553, 0.0038138605, 0.017545938, 0.017535519, 0.0066771028, -0.0056346566, -0.01041025, 0.008300587, 0.03595623, -0.015059399, -0.03659438, 0.017098658, -0.001846868, 0.018377855, 0.020022525, 0.032315258, 0.007804617, 0.017136335, -0.049641397, 0.0021616973, 0.01487886, 0.015920125, -0.012414415, 0.02732208, 0.039478082, 0.032164034, -0.055721737, -0.01589965, -0.055340573, -0.014910035, 0.019351715, 0.020711878, 0.06608685, -0.0042027673, 0.034433838, 0.019757664, -0.053433362, -0.013503944, 0.0017745359, -0.0021518487, -0.016649315, 0.056787595, 0.05485732, 0.011398314, 0.054610845, 0.025179526, -0.02694625, -0.042875834, 0.053497512, -0.0078063174, -0.032092623, 0.0015878901, 0.007234756, -0.005402673, 0.005059484, 0.074202895, -0.0012442522, -0.002525996, 0.010540503, 0.0045032348, 0.016989002, 0.007842403, -0.02031915, 0.026082495, 0.0009782778, -0.047594603, 0.023933053, -0.018158136, -0.023204053, -0.028291306, 0.02032343, 0.036438663, -0.0028127988, -0.025229894, 0.0048668603, -0.02134891, -0.04889245, -0.018371234, 0.014315416, 0.049745806, 0.018224802, 0.07054099, -0.07356177, 0.025843354, -0.03536464, 0.009958167, -0.011995419, 0.0011185072, 0.0044156304, -0.011416486, -0.011999269, -0.03607378, 0.05554972, -0.040670317, -0.008748484, -0.021122154, -0.026630286, -0.034994297, 0.04038684, 0.02512281, 0.024779445, 0.01734807, 0.026947048, 0.023414293, -0.018486926, -0.010824189, -0.015601604, 0.026508134, -0.010727221, -0.019428842, 0.0059344512, -0.02183771, 0.018732803, 0.061728053, -0.031089554, -0.0061309384, -0.012170976, 0.014639708, 0.06406931, -0.00021295605, 0.030456705, -0.03392049, 0.06332392, -0.0032031925, 0.019732384, -0.024752345, -0.012219408, 0.002401075, 0.00095222367, -0.040130466, -0.013230538, 0.024507403, 0.00038588303, -0.010628437, -0.020069892, -0.0091778645, -0.021861592, 0.04523284, 0.046188835, 0.03398155, -0.061485335, -0.01953832, 0.09637411, 0.059444495, -0.033719316, 0.012236311, 0.0025051418, -0.009695853, 0.040800422, -0.038275793, 0.005759768, 0.009360983, 0.011016493, -0.014367587, -0.020980302, -0.05621543, 0.0072761, 0.040003367, -0.0065101804, 0.0008521786, 0.05752916, 0.04087224, -0.021344045, 0.029359102, 0.056918085, 0.01481231, -0.065328546, -0.051711, 0.040324014, 0.039472487, 0.00028942068, -0.02890319, -0.017719701, 0.021527885, -0.010509229, 0.0049680467, 0.04182068, -0.006773699, 0.019639526, -0.036354255, 0.01438587, -0.021788953, 0.03663967, -0.050654884, -0.016374392, 0.01818318, 0.005170659, 0.017252646, -0.009644916, 0.009416437, -0.0050311713, 0.05899719, -0.015810044, -0.008406701, 0.066067785, -0.037994284, -0.011844192, -0.030280272, -8.6424225e-05, -0.026275268, 0.021008141, -0.07064195, 0.010534123, 0.012171037, 0.03873229, 0.031608045, -0.011524787, -0.018902369, 0.043874748, 0.026807146, 0.019843744, -0.014619262, 0.0083226245, 0.026949393, 0.034893837, 0.0055508506, -0.042768825, -0.024757575, -0.027095549, -0.01358015, -0.0075113825, 6.3796615e-05, 0.044898715, 0.02463679, 0.021851238, 0.017009713, 0.0056884023, -0.0012155358, -0.028812803, -0.025713231, 0.024948949, -0.051249746, 0.015030717, -0.027507978, 0.013987875, -0.035163946, 0.032785643, -0.028816937, -0.008973903, 0.0017470458, -0.0083494, 0.05761574, 0.02990923, -0.019372677, -0.018992485, 0.02550926, 0.013890094, -0.0040765987, -0.023219313, -0.00041008912, -0.004826704, 0.025466682, -0.095589615, -0.00040684832, 0.023361575, 0.04284614, -0.025602598, -0.05528899, -0.04531233, -0.013108013, -0.017266272, -0.045914996, -0.0009707767, -0.038953822, -0.020974245, 0.00018308398, 0.010956965, 0.021201702, -0.0014472457, -0.038396373, -0.0131336255, -0.011866877, -0.02444652, -0.029412333, 0.033053678, 0.0148148425, 0.017964901, -0.00059545884, -0.0057899924, -0.00039550714, -0.049381405, -0.02154011, -0.026475571, -0.04450135, 0.007300733, -0.028399253, -0.01040764, 0.031253666, -0.0075044436, 0.0061571323, 0.009239684, 0.038492963, -0.017560722, -0.041915014, 0.03748145, 0.0047262944, 0.009623691, -0.011705222, 0.0069222623, 0.014097556, 0.027454358, -0.03442649, -0.0016388226, 0.0529423, -0.016410822, 0.038686946, 0.003043477, -0.05237446, 0.0003470821, 0.034931436, -0.0038699878, -0.029861247, -0.04184306, -0.013839854, -0.017643664, -0.080214165, -0.0050084004, -0.02547889, -0.05458828, -0.016968917, 0.019724736, -0.016479434, -0.028994586, -0.015777733, 0.010934811, 0.013800441, -0.00035036265, -0.021450527, 0.008650378, 0.0036069527, 0.03003723, -0.017896602, -0.0060377903, 0.004396738, 0.021528073, -0.055637076, 0.024709955, -0.027978694, 0.008578263, 0.020510348, -0.010033129, -0.048869353, 0.011708439, -0.0120551, 0.0028068323, -0.05014475, 0.013385414, 0.02541199, -0.027935598, -0.007085062, 0.027451709, 0.02148373, -0.009953432, 0.011239228, -0.0069154906, -0.020488707, 0.016553875, -0.008458186, -0.062150426, -0.04859989, -0.04860158, 0.023738522, 0.027169395, 0.03358068, 0.03369936, -0.037197825, 0.038459804, -0.02807917, 0.03976932, -0.03608708, 0.04132699, -0.0064825052, 0.0352583, -0.016960079, -0.056628183, -0.03269243, 0.008086727, 0.106870726, -0.0029674836, -0.032134682, -0.013300067, -0.002038497, 0.0400676, -0.0044665495, 0.050639935, -0.028421132, -0.0032100338, 0.0051190127, 0.025391836, 0.006760539, 0.040972896, 0.033670194, -0.0037457005, -0.02953632, 0.04136543, -0.029028833, 0.046138186, 0.0054470054, 0.009241711, -0.031681012, -0.106729016, -0.015744135, -0.030006347, -0.018378872, -0.046579376, -0.009469625, 0.0011841267, 0.028496783, 0.010172226, -0.0023256368, -0.038266867, 0.03789038, 0.0025705348, 0.0108200405, 0.01278131, 0.06826698, 0.022966543, 0.027862687, 0.008836321, -0.06438112, -0.014712442, -0.02811046, 0.04462338, 0.019583343, 0.0041911122, -0.010450535, 0.0358305, 0.0074108574, -0.040468443, 0.0021544788, 0.011707619, 0.012094028, -0.06361027, 0.004477438, 0.0020871575, -0.07248526, -0.025801884, 0.028470084, -0.033508223, 0.023317544, -0.022429211, 0.021957127, 0.009301431, -0.0178207, 0.048027422, 0.005276454, 0.016097404, 0.0102554755, 0.042052805, 0.054792713, -0.006358971, 0.028359603, -0.06876384, 0.0035426312, -0.028322361, 0.03714794, 0.001778712, -0.04630417, -0.008128356, -0.03851808, 0.0044992585, -0.038789347, 0.00061308476, 0.033103563, -0.0005480606, -0.015730489, -0.015718184, 0.028014196, 0.054932453, 0.017180689, -0.02482305, 0.026789118, 0.051278494, 0.04395005, -0.0046009473, -0.012853065, 0.043463185, 0.022929287, -0.017515067, -0.0027633226, 0.010694646, -0.025438633, -0.07285533, -0.020091925, -0.0080963895, -0.05975945, -0.011413192, -0.0029255494, 0.031842347, -0.000713046, 0.027301496, -0.05517544, -0.06879526, -0.007458373, -0.04968724, -0.018447135, 0.0031384393, -0.011683095, 0.0039732503, -0.016069887, 0.025991568, 0.015502401, 0.00012176794, 0.011502548, 0.029929835, 0.028859463, -0.0347589, 0.02834919, -0.025514616, 0.093880735, 0.009537489, 0.035979234, -0.0073808786, -0.0039184005, 0.021177912, -0.045037366, -0.02465041, 0.01241995, -0.04430776, -0.004963603, 0.027810767, 0.03518751, 0.042041775, -0.06674767, 0.040487032, 0.006347213, 0.014919332, -0.049459815, 0.00719279, -0.0112957535, 0.038851503, -0.03412066, 0.10228839, 0.06328624, -0.029516064, 0.047730975, 0.0041117566, 0.018762277, -0.010183133, -0.025376698, -0.015922187, 0.010314202, -0.04871024, -0.0074796434, -0.015905295, -0.0016909653, -0.039035052, 0.010500884, -0.000748533, 0.040039998, 0.003301846, -0.02649019, 0.0034184977, -0.0057164878, -0.0073044184, -0.013703838, -0.013328431, 0.033882476, 0.059145197, -0.01548392, 0.04396831, -0.039770782, 0.024480386, -0.018274816, 0.04364667, -0.012514551, -0.01811829, 0.0053398274, -0.013561816, 0.059650913, 0.02360068, 0.010548052, -0.030777134, 0.0233678, 0.001806243, 0.0032160638, -0.016777178, 0.01966274, 0.051749073, 0.01278329, 0.051364396, 0.048698284, -0.050389525, 0.04392217, 0.04169097, 0.023618573, 0.039144065, 0.037507527, 0.05147729, 0.015287306, -0.053007204, -0.049763996, 0.05528352, 0.0376048, 0.032028094, -0.063992605, 0.03385273, 0.06943027, -0.026437588, -0.022515409, 0.017098168, 0.0065706535, 0.022878313, 0.04588939, -0.054177996, 0.01723673, -0.06473097, 0.017967718, 0.0013251797, -0.033139426, -0.0072945864, -0.010836204, -0.032635286, 0.03801997, 0.0031198116, 0.015583709, 0.023395047, -0.008145158, 0.023933342, -0.019637657, 0.0015128301, -0.013758694, -0.009423351, -0.023564149, 0.057373274, -0.007982398, 0.03354408, 0.016979218, -0.018186674, 0.04080128, 0.018534333, 0.023859257, 0.0031573926, -0.015217755, 0.0043780953, 0.021724468, -0.008114766, 0.009900437, 0.026897583, -0.041437134, 0.03290163, -0.048167463, -0.007174111, 0.03545826, -0.06208699, 0.014744109, 0.014470582, -0.014613295, 0.013811212, -0.025351983, -0.0468018, 0.007965019, 0.020791836, -0.009793674, -0.02550437, 0.0018169575, 0.033933606, -0.044916563, 0.04441153, -0.020923639, 0.05630165, -0.0031307503, -0.020926619, 0.0062865876, 0.0059055495, -0.01048605, 0.049237993, 0.06477576, -0.016144842, 0.013863138, 0.004353388, 0.010360109, -0.054498486, -0.029117454, 0.016099796, -0.061932363, 0.033399254, 0.040613323, -0.0034518973, 0.00675493, 0.0064320653, -0.016092768, -0.032148875, -0.0089891, 0.009424214, -0.0422774, 0.0082306415, -0.014798753, 0.0354656, -0.03426097, 0.0265262, -0.0057714437, 0.012938134, 0.058304805, 0.0012139766, 0.012661358, -0.008786471, -0.0176648, 0.040411226, 0.008020565, -0.01575593, 0.037214912, -0.029581644, 0.0005674357, 0.016098296, -0.013948431, -0.01119547, -0.033150014, 0.005167941, 0.01189257, -0.006237494, -0.043625575, 0.02619037, 0.035576534, 0.029012445, -0.028871007, -0.014601057, -0.05235596, 0.05800305, 0.02263392, -0.020325972, 0.009586974, 0.013099258, -0.03664665, -0.024749072, 0.011290492, 0.0070127635, -0.011284775, -0.022339016, 0.032303274, 0.03115308, -0.016450424, 0.01969099, 0.03997992, -0.0491729, -0.029670097]
     try:
         # Call the service method to get points with the specified vector
         results = await vector_db_service.get_points_with_vector(vector, vector_db_service.collection)
         return results
     except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))