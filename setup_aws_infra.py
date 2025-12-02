import boto3
import os
import time
import json
import sys

# Configuration
REGION = "eu-central-1"
PROJECT_NAME = "european-energy-forecasting"
BUCKET_NAME = "european-energy-forecasting-2025"
DYNAMODB_TABLE_NAME = "mlflow_metadata_table"
KEY_PAIR_NAME = f"{PROJECT_NAME}-key"

def setup_aws_infra():
    print("🚀 Starting AWS Infrastructure Setup...")
    
    # Check credentials
    try:
        sts = boto3.client('sts', region_name=REGION)
        identity = sts.get_caller_identity()
        print(f"✅ Authenticated as: {identity['Arn']}")
    except Exception as e:
        print("❌ AWS Credentials not found or invalid.")
        print(f"Error: {e}")
        sys.exit(1)

    # 1. Create S3 Bucket
    s3 = boto3.client('s3', region_name=REGION)
    try:
        # Check if bucket exists
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"✅ S3 Bucket already exists: {BUCKET_NAME}")
    except:
        try:
            if REGION == "us-east-1":
                s3.create_bucket(Bucket=BUCKET_NAME)
            else:
                s3.create_bucket(
                    Bucket=BUCKET_NAME,
                    CreateBucketConfiguration={'LocationConstraint': REGION}
                )
            print(f"✅ S3 Bucket created: {BUCKET_NAME}")
        except Exception as e:
            print(f"⚠️ S3 Bucket creation failed: {e}")

    # 2. Create Security Groups
    ec2 = boto3.client('ec2', region_name=REGION)
    sg_id = None
    try:
        vpcs = ec2.describe_vpcs()['Vpcs']
        if not vpcs:
            print("⚠️ No VPCs found. Attempting to create a Default VPC...")
            try:
                vpc_response = ec2.create_default_vpc()
                vpc_id = vpc_response['Vpc']['VpcId']
                print(f"✅ Default VPC created: {vpc_id}")
                # Wait for VPC to become available
                time.sleep(10) 
            except Exception as e:
                print(f"❌ Could not create default VPC: {e}")
                print("Please create a VPC in this region manually via AWS Console.")
                sys.exit(1)
        else:
            # Using the first available VPC
            vpc_id = vpcs[0]['VpcId']
            
        sg_name = f"{PROJECT_NAME}-mlflow-sg"
        
        try:
            # Check if SG exists
            sgs = ec2.describe_security_groups(Filters=[{'Name': 'group-name', 'Values': [sg_name]}])
            if sgs['SecurityGroups']:
                sg_id = sgs['SecurityGroups'][0]['GroupId']
                print(f"✅ Using existing Security Group: {sg_name} ({sg_id})")
            else:
                sg_response = ec2.create_security_group(
                    GroupName=sg_name,
                    Description="Allow MLflow traffic",
                    VpcId=vpc_id
                )
                sg_id = sg_response['GroupId']
                
                # Allow SSH (22) and MLflow (5000)
                ec2.authorize_security_group_ingress(
                    GroupId=sg_id,
                    IpPermissions=[
                        {'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]},
                        {'IpProtocol': 'tcp', 'FromPort': 5000, 'ToPort': 5000, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}
                    ]
                )
                print(f"✅ Security Group created: {sg_name} ({sg_id})")
        except Exception as e:
            print(f"⚠️ Security Group creation/check failed: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"⚠️ Security Group setup failed: {e}")
        sys.exit(1)

    # 3. Create Key Pair (for SSH)
    try:
        # Check if key exists
        ec2.describe_key_pairs(KeyNames=[KEY_PAIR_NAME])
        print(f"✅ Key Pair already exists: {KEY_PAIR_NAME}")
    except:
        try:
            key_pair = ec2.create_key_pair(KeyName=KEY_PAIR_NAME)
            with open(f"{KEY_PAIR_NAME}.pem", "w") as f:
                f.write(key_pair['KeyMaterial'])
            print(f"✅ Key Pair created: {KEY_PAIR_NAME}.pem (KEEP THIS SAFE!)")
        except Exception as e:
            print(f"⚠️ Key Pair creation failed: {e}")
            sys.exit(1)


    # 4. Launch EC2 Instance
    print("⏳ Launching EC2 Instance (t3.micro)...")
    try:
        # Ubuntu 22.04 LTS AMI ID for eu-central-1
        ami_response = ec2.describe_images(
            Owners=['099720109477'], # Canonical
            Filters=[
                {'Name': 'name', 'Values': ['ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*']},
                {'Name': 'state', 'Values': ['available']}
            ]
        )
        ami_id = sorted(ami_response['Images'], key=lambda x: x['CreationDate'], reverse=True)[0]['ImageId']
        
        instances = ec2.run_instances(
            ImageId=ami_id,
            InstanceType='t3.micro', # <-- FIX: Changed from t2.micro to t3.micro for Free Tier eligibility
            KeyName=KEY_PAIR_NAME,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[sg_id],
            TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': f'{PROJECT_NAME}-mlflow-server'}]}]
        )
        instance_id = instances['Instances'][0]['InstanceId']
        print(f"✅ EC2 Instance launched: {instance_id}")
    except Exception as e:
        print(f"⚠️ EC2 Launch failed: {e}")
        instance_id = "FAILED"
        sys.exit(1) # Exit if EC2 launch fails

    # 5. Create DynamoDB Table
    print(f"⏳ Creating DynamoDB Table ({DYNAMODB_TABLE_NAME})...")
    dynamodb = boto3.client('dynamodb', region_name=REGION)
    try:
        dynamodb.create_table(
            TableName=DYNAMODB_TABLE_NAME,
            KeySchema=[{'AttributeName': 'key', 'KeyType': 'HASH'}], # Partition key
            AttributeDefinitions=[{'AttributeName': 'key', 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST' # On-Demand (Free Tier eligible)
        )
        print(f"✅ DynamoDB Table created: {DYNAMODB_TABLE_NAME}")
    except Exception as e:
        if "ResourceInUseException" in str(e):
            print(f"✅ DynamoDB Table already exists: {DYNAMODB_TABLE_NAME}")
        else:
            print(f"⚠️ DynamoDB creation failed: {e}")
            sys.exit(1) # Exit if DynamoDB fails

    # 6. Get Public IP and Save details to file
    print("⏳ Waiting for EC2 instance to get a Public IP...")
    time.sleep(15) # Wait for IP to be assigned
    
    reservations = ec2.describe_instances(InstanceIds=[instance_id])['Reservations']
    public_ip = 'PENDING'
    if reservations and reservations[0]['Instances'][0].get('PublicIpAddress'):
        public_ip = reservations[0]['Instances'][0]['PublicIpAddress']
        print(f"✅ EC2 Public IP found: {public_ip}")
    else:
        print("⚠️ Could not retrieve public IP immediately. Check AWS Console.")

    config = {
        "bucket_name": BUCKET_NAME,
        "ec2_instance_id": instance_id,
        "ec2_public_ip": public_ip,
        "dynamodb_table": DYNAMODB_TABLE_NAME,
        "region": REGION,
        "key_pair_name": KEY_PAIR_NAME,
    }
    
    with open("aws_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("\n🎉 Infrastructure setup complete!")
    print(f"Config saved to 'aws_config.json'.")

if __name__ == "__main__":
    setup_aws_infra()