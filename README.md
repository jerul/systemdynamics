# systemdynamics
 Converting causal loop diagrams into computational system dynamics models

## Setup Instructions

### Install Dependencies

Make sure you have `pip` installed. Then, you can install the required packages using one of the following methods:

#### Method 1: Using `requirements.txt`

```sh
pip install -r requirements.txt
```

#### Method 2: Using `setup.py`
```sh
pip install .
```

### Running the Package
To run the package, from the Examples directory, use the following command:
```sh
cd path
python initialize.py <setting_name>
```

Replace <setting_name> with the appropriate setting, such as 'Sleep'.

### Example
Here is an example of how you might run the initialization script:

```sh
cd user/systemdynamics
python initialize.py Sleep 
```

Make sure to adjust the path to Examples based on your directory structure and that initialize.py is the same directory as the Examples folder.

## Additional Information

- If you encounter any issues, please ensure that you have all the necessary dependencies installed.
- For more information, refer to the [documentation]() or [contact us]().